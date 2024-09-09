import functions
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import argparse
from PIL import Image
import time

parser = argparse.ArgumentParser(description='Python script')

#####################################
#           HYPERPARAMETERS         #
#####################################

# Path to netG.pth
parser.add_argument("--netG_path", type=str, default="netG.pth", help="Path to the pretrained weights")

# Path to save
parser.add_argument("--destination_path", type=str, default=os.getcwd(), help="Path where to save the images")

parser.add_argument('--clear', action='store_true', help="Flag. If included clear the directory.")

# Image size
parser.add_argument("--image_size", type=int, default=128, help="Image size")

# ID of GPU
parser.add_argument("--gpu", type=int, default=0, help="Index of the GPU")

# Size of z latent vector (i.e. size of generator input)
parser.add_argument("--nz", type=int, default=256, help="Size of latent dimension (generator input)")

# Number of images to generate
parser.add_argument("--num_samples", type=int, default=32, help="Number of images to generate per class")

# Negative slope for LeakyReLU
parser.add_argument("--slope", type=float, default=0.01, help="Negative slope for LeakyReLU. (0.01)")

# Batch size for generation
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image generation")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#################################
#           GENERATION          #
#################################

# Record the start time
total_start_time = time.time()

# Image Generation Loop
Generator = functions.UpsampleGenerator(args.nz, 64, 6, 5, args.image_size, 0.1, args.slope).to(device)
Generator.load_state_dict(torch.load(args.netG_path, map_location=device))
Generator.eval()


if args.clear:
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(args.destination_path):
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(root, file))

# Unnormalize function
unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

# Create subfolders for each class in "pressure" and "velocity" folders
for i in range(5):
    os.makedirs(f"{args.destination_path}/pressure/class_{i}", exist_ok=True)
    os.makedirs(f"{args.destination_path}/velocity/class_{i}", exist_ok=True)

# Generate and save images for each class
num_images_per_class = args.num_samples

for class_label in range(5):
    # Calculate the number of batches required (last batch could be a remainder)
    num_batches = (num_images_per_class + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, num_images_per_class - batch_idx * args.batch_size)
        
        # Generate a batch of latent vectors (random noise)
        noise = torch.randn(current_batch_size, args.nz, device=device, dtype=torch.float)
        # Create a tensor of the current class labels
        labels = torch.full((current_batch_size,), class_label, device=device, dtype=torch.long)

        # Generate a batch of fake images with the Generator
        with torch.no_grad():
            fake_images = Generator(noise, labels).detach().cpu()

        # Separate and unnormalize the images from each dataset
        for i, fake_image in enumerate(fake_images):
            # Split the channels: first 3 for the first dataset, last 3 for the second dataset
            fake_pressure = fake_image[:3, :, :]
            fake_velocity = fake_image[3:, :, :]

            # Unnormalize each part
            fake_pressure = unnormalize(fake_pressure)
            fake_velocity = unnormalize(fake_velocity)

            # Convert to numpy and save each part separately
            fake_pressure = fake_pressure.numpy()
            fake_pressure = np.transpose(fake_pressure, (1, 2, 0))  # Change from CHW to HWC
            fake_pressure = Image.fromarray((fake_pressure * 255).astype(np.uint8))

            fake_velocity = fake_velocity.numpy()
            fake_velocity = np.transpose(fake_velocity, (1, 2, 0))  # Change from CHW to HWC
            fake_velocity = Image.fromarray((fake_velocity * 255).astype(np.uint8))

            # Save to disk
            fake_pressure.save(f"{args.destination_path}/pressure/class_{class_label}/pressure_{batch_idx * args.batch_size + i}.png")
            fake_velocity.save(f"{args.destination_path}/velocity/class_{class_label}/velocity_{batch_idx * args.batch_size + i}.png")

print(f"Generated images saved to {args.destination_path}")

# Record the end time
total_end_time = time.time()

# Calculate the total time taken
total_time = total_end_time - total_start_time

# Convert the total time to hours, minutes, and seconds
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

# Print the total time taken
print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")