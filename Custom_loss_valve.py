import torch
import torch.nn.functional as F


class Constraint:
    def __init__(self, image_tensor, image_size, colormap_tensor):
        self.image_size = image_size
        self.colormap_tensor = colormap_tensor
        
        # Split the 6-channel image into two 3-channel images
        self.img_pressure = image_tensor[:, :3, :, :]  # First 3 channels
        self.img_velocity = image_tensor[:, 3:, :, :]  # Last 3 channels
        
        # Convert to grayscale
        self.img_gray_pressure = 0.2989 * self.img_pressure[:, 0, :, :] + 0.5870 * self.img_pressure[:, 1, :, :] + 0.1140 * self.img_pressure[:, 2, :, :]
        self.img_gray_velocity = 0.2989 * self.img_velocity[:, 0, :, :] + 0.5870 * self.img_velocity[:, 1, :, :] + 0.1140 * self.img_velocity[:, 2, :, :]
        
        # Process all images in the batch
        self.flags_pressure, self.horiz_lines_pressure, self.gap_lines_pressure, self.vert_lines_pressure = self.process_batch(self.img_gray_pressure)
        self.flags_velocity, self.horiz_lines_velocity, self.gap_lines_velocity, self.vert_lines_velocity = self.process_batch(self.img_gray_velocity)

    def process_batch(self, img_gray_batch):
        flags, horiz_lines_batch, gap_lines_batch, vert_lines_batch = [], [], [], []
        for img_gray in img_gray_batch:
            flag, horiz_lines, gap_lines, vert_lines = self.find_lines(img_gray)
            flags.append(flag)
            horiz_lines_batch.append(horiz_lines)
            gap_lines_batch.append(gap_lines)
            vert_lines_batch.append(vert_lines)

        return flags, horiz_lines_batch, gap_lines_batch, vert_lines_batch

    def find_lines(self, img_gray):
        # Use Sobel operator to detect edges
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(img_gray.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(img_gray.device)

        edges_x = F.conv2d(img_gray.unsqueeze(0).unsqueeze(0), sobel_x, padding=1).squeeze(0).squeeze(0)
        edges_y = F.conv2d(img_gray.unsqueeze(0).unsqueeze(0), sobel_y, padding=1).squeeze(0).squeeze(0)

        edges = torch.sqrt(edges_x**2 + edges_y**2)

        # Threshold edges to get binary image
        edges_bin = (edges > edges.mean()).float()

        # Morphological operations to find horizontal and vertical lines
        horiz_kernel = torch.ones((1, 50), device=img_gray.device).unsqueeze(0).unsqueeze(0)
        vert_kernel = torch.ones((50, 1), device=img_gray.device).unsqueeze(0).unsqueeze(0)

        horiz_lines = F.conv2d(edges_bin.unsqueeze(0).unsqueeze(0), horiz_kernel, padding=(0, 10)).squeeze(0).squeeze(0).max(dim=1).values
        vert_lines = F.conv2d(edges_bin.unsqueeze(0).unsqueeze(0), vert_kernel, padding=(10, 0)).squeeze(0).squeeze(0).max(dim=0).values

        horiz_lines = (horiz_lines > 47).nonzero(as_tuple=True)
        vert_lines = (vert_lines > 47).nonzero(as_tuple=True)

        horiz_lines = [torch.tensor([0, y, self.image_size - 1, y], dtype=torch.float32) for y in horiz_lines[0]]
        vert_lines = [torch.tensor([x, 0, x, self.image_size - 1], dtype=torch.float32) for x in vert_lines[0]]

        # Filter lines to avoid duplicate detections
        horiz_lines = self.filter_lines(horiz_lines, axis=1, threshold=5)
        vert_lines = self.filter_lines(vert_lines, axis=0, threshold=8)

        # Validate the detected lines
        if len(horiz_lines) == 7 and len(vert_lines) == 6:
            horiz_lines.sort(key=lambda line: line[1])
            vert_lines.sort(key=lambda line: line[0])
            gap_lines = [horiz_lines[2], horiz_lines[3]]
            horiz_lines = [horiz_lines[1], horiz_lines[4], horiz_lines[5]]
            vert_lines = [vert_lines[1], vert_lines[2], vert_lines[3], vert_lines[4]]
            flag = True
        else:
            gap_lines = []
            flag = False

        return flag, horiz_lines, gap_lines, vert_lines

    def filter_lines(self, lines, axis, threshold):
        lines.sort(key=lambda line: line[axis])

        filtered_lines = []
        prev_line = None
        for line in lines:
            if prev_line is None or torch.abs(line[axis] - prev_line[axis]) > threshold:
                filtered_lines.append(line)
                prev_line = line

        return filtered_lines

    def differentiable_argmin(self, tensor, temperature=0.0001):
        """
        Compute a differentiable approximation of the argmin using softmin with a temperature parameter.
        """
        scaled_tensor = tensor / temperature
        weights = torch.nn.functional.softmin(scaled_tensor, dim=0)
        indices = torch.arange(tensor.size(0), device=tensor.device, dtype=torch.float32)
        soft_argmin = torch.sum(weights * indices)

        return soft_argmin

    def RGB_to_Pa(self, color):
        '''
        Map the RGB values to Pascal in the correct empirical range, using viridis colormap
        '''
        distances = torch.sqrt(torch.sum(torch.square(self.colormap_tensor - color), dim=1))
        soft_argmin = self.differentiable_argmin(distances)
        normalized_value = soft_argmin / (self.colormap_tensor.size(0) - 1)

        min_P, max_P = -67783288.0, 18878706.0
        pressure = normalized_value * (max_P - min_P) + min_P

        return pressure

    def RGB_to_velocity(self, color):
        '''
        Map the RGB values to meters/second in the correct empirical range, using viridis colormap
        '''
        distances = torch.sqrt(torch.sum(torch.square(self.colormap_tensor - color), dim=1))
        soft_argmin = self.differentiable_argmin(distances)
        normalized_value = soft_argmin / (self.colormap_tensor.size(0) - 1)

        min_V, max_V = 2.5215445e-06, 368.293823
        velocity = normalized_value * (max_V - min_V) + min_V

        return velocity

    def get_gap(self, index, twenty_degrees_flag, device):
        '''
        Find the gap measure and convert to meters
        '''
        if twenty_degrees_flag:
            if self.flags_pressure[index]:
                gap_pixel = self.gap_lines_pressure[index][1][1] - self.gap_lines_pressure[index][0][1]
                gap_m = torch.pow((gap_pixel / 20.28413226), 1 / 0.61672483) / 1000
            elif self.flags_velocity[index]:
                gap_pixel = self.gap_lines_velocity[index][1][1] - self.gap_lines_velocity[index][0][1]
                gap_m = torch.pow((gap_pixel / 20.28413226), 1 / 0.61672483) / 1000
            else:
                gap_m = torch.tensor(0.0)
        else:
            if self.flags_pressure[index]:
                gap_pixel = self.gap_lines_pressure[index][1][1] - self.gap_lines_pressure[index][0][1]
                gap_m = torch.pow((gap_pixel / 15.65135632), 1 / 0.49930509) / 1000
            elif self.flags_velocity[index]:
                gap_pixel = self.gap_lines_velocity[index][1][1] - self.gap_lines_velocity[index][0][1]
                gap_m = torch.pow((gap_pixel / 15.65135632), 1 / 0.49930509) / 1000
            else:
                gap_m = torch.tensor(0.0)
            if gap_m.item() > 0.00022 and gap_m.item() < 0.0003:
                gap_m = torch.tensor(0.00028, device=device, dtype=torch.float32)

        return gap_m

    def get_quantities(self, index):
        '''
        Find the pressure in inlet and outlet and velocity before and after the gap
        '''
        if self.flags_pressure[index]:
            # Find the meaningful lines for pressure detection
            y_outlet = int(self.horiz_lines_pressure[index][0][1])
            y_inlet = int(self.horiz_lines_pressure[index][2][1])
            left_outlet, right_outlet = int(self.vert_lines_pressure[index][0][0]), int(self.vert_lines_pressure[index][1][0])
            left_inlet, right_inlet = int(self.vert_lines_pressure[index][2][0]), int(self.vert_lines_pressure[index][3][0])
            # Find the meaningful lines for velocity detection
            upper_gap = int(self.gap_lines_pressure[index][0][1])
            lower_gap = int(self.gap_lines_pressure[index][1][1])
        elif self.flags_velocity[index]:
            # Find the meaningful lines for pressure detection
            y_outlet = int(self.horiz_lines_velocity[index][0][1])
            y_inlet = int(self.horiz_lines_velocity[index][2][1])
            left_outlet, right_outlet = int(self.vert_lines_velocity[index][0][0]), int(self.vert_lines_velocity[index][1][0])
            left_inlet, right_inlet = int(self.vert_lines_velocity[index][2][0]), int(self.vert_lines_velocity[index][3][0])
            # Find the meaningful lines for velocity detection
            upper_gap = int(self.gap_lines_velocity[index][0][1])
            lower_gap = int(self.gap_lines_velocity[index][1][1])
        else:
            # Return high loss values to signify error
            return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(False)

        vert_tol = 8
        horiz_tol = 8

        # Define a threshold for determining "white" pixels
        white_threshold = 0.98  # Adjust this threshold as needed

        # Pixel RGB of pressure and velocity
        outlet_pressure = self.img_pressure[index, :, y_outlet + vert_tol, left_outlet + horiz_tol:right_outlet - horiz_tol+ 1]
        inlet_pressure = self.img_pressure[index, :, y_inlet - vert_tol, left_inlet + horiz_tol:right_inlet - horiz_tol + 1]

        pixel_color_twenty_degrees = self.img_pressure[:, :, 137, 170]
        if torch.all(pixel_color_twenty_degrees >= torch.tensor([white_threshold, white_threshold, white_threshold], device=self.img_gray_pressure.device)): # 45 degrees
            twenty_degrees_flag = False
            inlet_velocity = self.img_velocity[index, :, upper_gap+2:lower_gap-1, 172]
        else: # 20 degrees
            twenty_degrees_flag = True
            inlet_velocity = self.img_velocity[index, :, upper_gap+2:lower_gap-1, 154]
    
        outlet_velocity = self.img_velocity[index, :, upper_gap+2:lower_gap-1, 121]

        # Mean RGB of pressure and velocity after excluding white pixels
        outlet_mean_pressure = torch.mean(outlet_pressure.float(), dim=1) * 0.5 + 0.5
        inlet_mean_pressure = torch.mean(inlet_pressure.float(), dim=1) * 0.5 + 0.5

        outlet_mean_velocity = torch.mean(outlet_velocity.float(), dim=1) * 0.5 + 0.5
        inlet_mean_velocity = torch.mean(inlet_velocity.float(), dim=1) * 0.5 + 0.5

        # Convert to Pascal and m/s
        outlet_pressure = self.RGB_to_Pa(outlet_mean_pressure)
        inlet_pressure = self.RGB_to_Pa(inlet_mean_pressure)

        outlet_velocity = self.RGB_to_velocity(outlet_mean_velocity)
        inlet_velocity = self.RGB_to_velocity(inlet_mean_velocity)

        return inlet_pressure, outlet_pressure, inlet_velocity, outlet_velocity, twenty_degrees_flag

    
def Mass_loss(inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag):
    outlet_mass = 0.01207 * outlet_velocity * gap_height
    if twenty_degrees_flag:
        inlet_mass = 0.00863 * inlet_velocity * gap_height
    else:
        inlet_mass = 0.007 * inlet_velocity * gap_height

    # Scaling factor tuned for each gap differently
    if twenty_degrees_flag:
        if gap_height.item() > 0.00045:
            scale_factor = 1.3e10
        elif gap_height.item() > 0.00034 and gap_height.item() <= 0.00045:
            scale_factor = 2.4e9
        elif gap_height.item() >= 0.00025 and gap_height.item() <= 0.00034:
            scale_factor = 2.7e9
        elif gap_height.item() < 0.00025:
            scale_factor = 3.7e8
    else:
        if gap_height.item() > 0.00045:
            scale_factor = 3.7e9
        elif gap_height.item() > 0.00034 and gap_height.item() <= 0.00045:
            scale_factor = 2.1e8
        elif gap_height.item() > 0.00025 and gap_height.item() <= 0.00034:
            scale_factor = 1.5e12
        elif gap_height.item() > 0.00017 and gap_height.item() <= 0.00025:
            scale_factor = 8e8
        elif gap_height.item() <= 0.00017:
            scale_factor = 6e7
    
    return torch.pow(inlet_mass - outlet_mass, 2) * scale_factor

def Reynolds(velocity, gap, rho=997, mu=0.001):
    return (rho * velocity * gap) / (mu)

def K_in(Reynolds_in, twenty_degrees_flag):
    '''
    Gap inlet friction coefficient, valid for all Reynolds
    '''
    k_in = 5.85 * torch.pow(Reynolds_in, -0.56)
    if twenty_degrees_flag:
        k_in = k_in * (1.34 - 5.26 * 10 ** (-4) * 20 ** 2 + 6.19 * 10 ** (-6) * 20 ** 3)
    return k_in

def K_out(Reynolds_out):
    '''
    Gap exit friction coefficient, with threshold Re at 1000
    '''
    return 1.95 * torch.pow(Reynolds_out, -0.127) if Reynolds_out <= 1000 else 0.598 * torch.pow(Reynolds_out, 0.038)

def K_gap(Reynolds_out, twenty_degrees_flag):
    '''
    Gap friction coefficient, with threshold Re at 2000
    '''
    k_gap = 9.94 * torch.pow(Reynolds_out, -0.243) if Reynolds_out <= 2000 else 1.57
    if twenty_degrees_flag:
        k_gap = k_gap * (0.729 + 1.45 * 10 ** (-4) * 20 ** 2)
    return k_gap

def friction(Reynolds_out):
    '''
    Moody/Fanning friction factor, with threshold Re at 4000
    '''
    return 64/Reynolds_out if Reynolds_out < 4000 else torch.pow(-1.8 * torch.log10((6.9 / Reynolds_out)), -2)

def Pressure_losses(inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag, rho=997):
    k_in = K_in(Reynolds(inlet_velocity, gap_height), twenty_degrees_flag)
    k_out = K_out(Reynolds(outlet_velocity, gap_height))
    k_gap = K_gap(Reynolds(outlet_velocity, gap_height), twenty_degrees_flag)
    f = friction(Reynolds(outlet_velocity, gap_height))
    if twenty_degrees_flag:
        L_g = 0.00344
    else:
        L_g = 0.00507
    delta_P = (rho * k_in * inlet_velocity ** 2) / 2 + (rho * k_gap * f * L_g * outlet_velocity ** 2) / gap_height + (rho * k_out * outlet_velocity ** 2) / 2

    return delta_P

def Pressure_balance_loss(inlet_pressure, outlet_pressure, inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag):
    delta_p = Pressure_losses(inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag) 

    total_outlet_pressure = delta_p + outlet_pressure

    # Scaling factor tuned for each gap differently
    if twenty_degrees_flag:
        if gap_height.item() > 0.00045:
            scale_factor = 5e10
        elif gap_height.item() > 0.00034 and gap_height.item() <= 0.00045:
            scale_factor = 4e11
        elif gap_height.item() >= 0.00025 and gap_height.item() <= 0.00034:
            scale_factor = 3e11
        elif gap_height.item() < 0.00025:
            scale_factor = 4e12
    else:
        if gap_height.item() > 0.00045:
            scale_factor = 2e11
        elif gap_height.item() > 0.00034 and gap_height.item() <= 0.00045:
            scale_factor = 2.5e11
        elif gap_height.item() > 0.00025 and gap_height.item() <= 0.00034:
            scale_factor = 1.2e12
        elif gap_height.item() > 0.00017 and gap_height.item() <= 0.00025:
            scale_factor = 1.5e12
        elif gap_height.item() <= 0.00017:
            scale_factor = 3.5e13

    return torch.pow(inlet_pressure - total_outlet_pressure, 2) / scale_factor

def compute_losses(image_tensor, image_size, colormap_tensor):
    constraint = Constraint(image_tensor, image_size, colormap_tensor)
    total_mass_loss = 0
    total_pressure_loss = 0

    for i in range(image_tensor.size(0)):
        inlet_pressure, outlet_pressure, inlet_velocity, outlet_velocity, twenty_degrees_flag = constraint.get_quantities(i)
        gap_height = constraint.get_gap(i, twenty_degrees_flag, image_tensor.device)

        mass_loss = Mass_loss(inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag)
        pressure_loss = Pressure_balance_loss(inlet_pressure, outlet_pressure, inlet_velocity, outlet_velocity, gap_height, twenty_degrees_flag)
        
        # Skip the computation if the values indicate an error (e.g., inf)
        if torch.isinf(inlet_pressure) or torch.isinf(outlet_pressure) or torch.isinf(inlet_velocity) or torch.isinf(outlet_velocity):
            return torch.tensor(0.0), torch.tensor(0.0)

        total_mass_loss += mass_loss
        total_pressure_loss += pressure_loss

    batch_size = image_tensor.size(0) if image_tensor.size(0) > 0 else 1  # Avoid division by zero

    average_mass_loss = total_mass_loss / batch_size
    average_pressure_loss = total_pressure_loss / batch_size

    return average_mass_loss, average_pressure_loss