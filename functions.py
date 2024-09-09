import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


########################################
#           DATASET CLASSES            #
########################################



class PairedImageDataset(Dataset):
    '''
    Load coupled datasets from two directories with subdirectories representing classes.
    Concatenate the images to form a 6-channel image. The labels are derived from the subdirectory names.
    '''
    def __init__(self, dataset1_dir, dataset2_dir, transform=None):
        """
        Args:
            dataset1_dir (string): Root directory with all the images from the first dataset, organized in class subdirectories.
            dataset2_dir (string): Root directory with all the images from the second dataset, organized in class subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset1_dir = dataset1_dir
        self.dataset2_dir = dataset2_dir
        self.transform = transform
        
        # Obtain the class names from subdirectory names, sorted alphabetically
        self.classes = sorted(os.listdir(dataset1_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Create a list of (image1_path, image2_path, label) tuples
        self.samples = []
        for cls_name in self.classes:
            class_index = self.class_to_idx[cls_name]
            class_dir1 = os.path.join(dataset1_dir, cls_name)
            class_dir2 = os.path.join(dataset2_dir, cls_name)
            if os.path.isdir(class_dir1) and os.path.isdir(class_dir2):
                for img_name in sorted(os.listdir(class_dir1)):
                    img_path1 = os.path.join(class_dir1, img_name)
                    img_path2 = os.path.join(class_dir2, img_name)
                    if os.path.isfile(img_path1) and os.path.isfile(img_path2):
                        self.samples.append((img_path1, img_path2, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path1, img_path2, label = self.samples[idx]
        
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        # Concatenate images along the channel dimension
        combined_image = torch.cat((image1, image2), dim=0)
        
        return combined_image, label


####################################
#           GAN CLASSES            #
####################################


class TConvGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes, image_size, dropout, slope):
        super(TConvGenerator, self).__init__()
        assert image_size % 16 == 0, "Image size has to be a multiple of 16"
        self.n_classes = n_classes
        layers = []
        mult = image_size // 8
        # input is Z, going into a convolution
        layers += [
            nn.ConvTranspose2d(nz + n_classes, ngf * mult, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * mult, affine=True),
            nn.LeakyReLU(slope, inplace=True),
            nn.Dropout(dropout)
        ]
        while mult > 1:
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * (mult // 2), 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ngf * (mult // 2), affine=True),
                nn.LeakyReLU(slope, inplace=True),
                nn.Dropout(dropout)
            ]
            mult //= 2
        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, input, class_labels):
        class_labels = nn.functional.one_hot(class_labels, num_classes=self.n_classes).float()
        class_labels = class_labels.view(-1, self.n_classes, 1, 1)
        input = torch.cat([input, class_labels], 1)
        return self.main(input)
    
class UpsampleGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes, image_size, dropout, slope):
        super(UpsampleGenerator, self).__init__()
        assert image_size % 16 == 0, "Image size has to be a multiple of 16"
        self.n_classes = n_classes
        self.ngf = ngf
        self.mult = image_size // 8
        self.fc = nn.Sequential(
            nn.Linear(nz + n_classes, ngf * self.mult * 4 * 4, bias=False),
            nn.LayerNorm(ngf * self.mult * 4 * 4, elementwise_affine=True),
            nn.LeakyReLU(slope, inplace=True),
            nn.Dropout(dropout)
        )
        layers = []
        mult = self.mult
        while (mult > 1):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ngf * mult, ngf * (mult // 2), 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * (mult // 2), affine=True),
                nn.LeakyReLU(slope, inplace=True),
                nn.Dropout(dropout)
            ]
            mult //= 2
        layers += [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input, class_labels):
        batch_size = input.size(0)
        class_labels_one_hot = torch.zeros(batch_size, self.n_classes, device=input.device)
        class_labels_one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
        input = torch.cat([input, class_labels_one_hot], 1)
        input = input.view(input.size(0), -1)
        output = self.fc(input)
        output = output.view(-1, self.ngf * self.mult, 4, 4)
        return self.main(output)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_classes, image_size, model, noise, dropout, slope):
        super(Discriminator, self).__init__()
        assert image_size % 16 == 0, "Image size has to be a multiple of 16"
        self.n_classes = n_classes
        self.noise = noise
        layers = []
        mult = 1
        layers += [nn.Conv2d(nc + n_classes, ndf * mult, 4, 2, 1, bias=False), 
                   nn.LeakyReLU(slope, inplace=True),
                   nn.Dropout(dropout)]
        while mult < image_size // 8:
            layers += [
                nn.Conv2d(ndf * mult, ndf * (mult * 2), 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ndf * (mult * 2), affine=True),
                nn.LeakyReLU(slope, inplace=True),
                nn.Dropout(dropout)
            ]
            mult *= 2
        if model == "GAN": layers += [nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=True), nn.Sigmoid()]
        elif model == "WGAN-GP": layers += [nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=True)]
        self.main = nn.Sequential(*layers)

    def forward(self, input, class_labels):
        batch_size = input.size(0)
        class_labels_one_hot = torch.zeros(batch_size, self.n_classes, device=input.device)
        class_labels_one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
        class_labels_one_hot = class_labels_one_hot.view(batch_size, self.n_classes, 1, 1)
        class_labels_one_hot = class_labels_one_hot.expand(-1, -1, input.shape[2], input.shape[3])
        input = torch.cat([input, class_labels_one_hot], 1)
        n = torch.randn_like(input) * self.noise
        return self.main(input + n)


####################################
#           GAN FUNCTIONS          #
####################################


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


####################################
#               OTHERS             #
####################################


def custom_colormap(device):
    # Define the number of levels you want
    num_levels = 2**14  # Increase this number for more unique colors

    # Define the colors for the custom colormap using more interpolation points
    viridis = plt.get_cmap('viridis')
    base_colors = viridis(np.linspace(0, 1, 256))  # Base colors from viridis
    
    # Interpolate to create more unique colors
    interpolated_colors = np.zeros((num_levels, 4))
    for i in range(4):
        interpolated_colors[:, i] = np.interp(np.linspace(0, 1, num_levels), np.linspace(0, 1, 256), base_colors[:, i])

    # Create a custom colormap with more unique colors
    new_cmap = LinearSegmentedColormap.from_list('extended_viridis', interpolated_colors, N=num_levels)

    # Extract the colors and ignore the alpha channel
    colormap_array = new_cmap(np.linspace(0, 1, num_levels))[:, :3]

    # Convert the colormap array to a PyTorch tensor
    colormap_tensor_extended = torch.tensor(colormap_array, dtype=torch.float32, device=device)

    return colormap_tensor_extended


def experience_replay(replay_buffer, fake, class_labels, batch_size, prob):
    # Store a copy of the fake images and class labels into the replay buffer
    # Only if they are of full batch size (because last batch could be smaller)
    if fake.shape[0] == batch_size:
        if len(replay_buffer) < batch_size:
            replay_buffer.append((fake.detach().cpu(), class_labels.detach().cpu()))
        else:
            # If the buffer is full, remove the oldest image and add the new one
            replay_buffer.pop(0)
            replay_buffer.append((fake.detach().cpu(), class_labels.detach().cpu()))

    # Filter the replay buffer to include only images of the same class
    same_class_replay_buffer = [item for item in replay_buffer if item[1].item() == class_labels.item()]

    # Decide whether to use a random previous fake image of the same class or the current one
    if len(same_class_replay_buffer) >= batch_size and torch.rand(1).item() < prob:
        index = int(torch.rand(1).item() * len(same_class_replay_buffer))
        fake_input, class_label_input = same_class_replay_buffer[index]
        fake_input, class_label_input = fake_input.to(fake.device), class_label_input.to(class_labels.device)
    else:
        fake_input, class_label_input = fake, class_labels

    return fake_input, class_label_input, replay_buffer


def save_generated_images(generator, num_classes, fixed_noise, device, output_dir):
    '''
    Save images each epoch with titles using matplotlib, but processing images as in the post-training script.
    '''
    generator.eval()

    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )

    plt.figure(figsize=(8, 2 * num_classes))

    for class_idx in range(num_classes):
        # Create a tensor for the class index
        label = torch.tensor([class_idx], dtype=torch.long, device=device)

        # Generate image for the class
        with torch.no_grad():
            fake_image = generator(fixed_noise, label).detach().cpu()

        # Extract the image for plotting
        image = fake_image[0]

        # Separe pressure to velocity
        pressure_img = image[:3, :, :]
        velocity_img = image[3:, :, :]

        # Unnormalize images (align with post-training script)
        pressure_img = unnormalize(pressure_img)
        velocity_img = unnormalize(velocity_img)

        # Convert to numpy for plotting
        pressure_img = pressure_img.numpy()
        velocity_img = velocity_img.numpy()

        # Convert from CHW to HWC
        pressure_img = np.transpose(pressure_img, (1, 2, 0))
        velocity_img = np.transpose(velocity_img, (1, 2, 0))

        # Convert images to uint8 format for better quality display
        pressure_img = Image.fromarray((pressure_img * 255).astype(np.uint8))
        velocity_img = Image.fromarray((velocity_img * 255).astype(np.uint8))

        # Plot pressure image
        ax = plt.subplot(num_classes, 2, class_idx * 2 + 1)
        ax.imshow(pressure_img)
        ax.set_title(f'Class {class_idx} - Pressure')
        ax.axis('off')

        # Plot velocity image
        ax = plt.subplot(num_classes, 2, class_idx * 2 + 2)
        ax.imshow(velocity_img)
        ax.set_title(f'Class {class_idx} - Velocity')
        ax.axis('off')

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()
    generator.train()

def save_separated_generated_images(generator, fixed_noise, device, output_dir, epoch):
    '''
    Save separate images for pressure and velocity for class 0 using matplotlib, 
    and process images as in the post-training script.
    '''
    generator.eval()

    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )

    # Generate image for class 0 only
    class_idx = 0

    # Create a tensor for the class index
    label = torch.tensor([class_idx], dtype=torch.long, device=device)

    # Generate image for the class
    with torch.no_grad():
        fake_image = generator(fixed_noise, label).detach().cpu()

    # Extract the image for plotting
    image = fake_image[0]

    # Separate pressure and velocity
    pressure_img = image[:3, :, :]
    velocity_img = image[3:, :, :]

    # Unnormalize images (align with post-training script)
    pressure_img = unnormalize(pressure_img)
    velocity_img = unnormalize(velocity_img)

    # Convert to numpy for plotting
    pressure_img = pressure_img.numpy()
    velocity_img = velocity_img.numpy()

    # Convert from CHW to HWC
    pressure_img = np.transpose(pressure_img, (1, 2, 0))
    velocity_img = np.transpose(velocity_img, (1, 2, 0))

    # Convert images to uint8 format for better quality display
    pressure_img = Image.fromarray((pressure_img * 255).astype(np.uint8))
    velocity_img = Image.fromarray((velocity_img * 255).astype(np.uint8))

    # Plot and save pressure image
    plt.figure(figsize=(4, 4))
    plt.imshow(pressure_img)
    plt.axis('off')
    pressure_path = f'{output_dir}/pressure_epoch_{epoch}.png'
    plt.savefig(pressure_path, bbox_inches="tight")
    plt.close()

    # Plot and save velocity image
    plt.figure(figsize=(4, 4))
    plt.imshow(velocity_img)
    plt.axis('off')
    velocity_path = f'{output_dir}/velocity_epoch_{epoch}.png'
    plt.savefig(velocity_path, bbox_inches="tight")
    plt.close()

    generator.train()