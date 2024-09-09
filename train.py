#############################
#           IMPORT          #
#############################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time
import glob
import argparse
import functions
import Custom_loss_valve as ValveLoss
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torch.cuda.empty_cache() 

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


######################################
#           HYPERPARAMETERS          #
######################################


parser = argparse.ArgumentParser(description='Python script to train PI-cGAN or PI-cWGAN')

# Root directory for dataset
parser.add_argument("--dataset_pressure", type=str, default=os.getcwd(), help="Path to the dataset of the pressures")
parser.add_argument("--dataset_velocity", type=str, default=os.getcwd(), help="Path to the dataset of the velocities")
parser.add_argument("--dataroot", type=str, default=os.getcwd(), help="Path to the dataset of the square dataset")
parser.add_argument("--dataset", type=str, default="lineare", help="Type of dataset, use to name the losses")

# Labels
parser.add_argument("--real_label", type=float, default=1., help="Label of real images. (1)")
parser.add_argument("--fake_label", type=float, default=0., help="Label of fake images. (0)")

# Model to use
parser.add_argument('--model', type=str, default='GAN', help='Choose the model ["GAN", "WGAN-GP"]. ("GAN")')

# Type of generator to use
parser.add_argument('--generator_model', type=str, default='TConv', help='Choose the generator class ["TConv", "Upsampling"]. ("TConv")')

# Number of workers for dataloader
parser.add_argument("--workers", type=int, default=10, help="Number of workers for the Dataloader. (10)")

# Batch size during training
parser.add_argument("--batch_size", type=int, default=16, help="Batch size. (16)")

# Spatial size of training images. All images will be resized to this
# size using a transformer.
parser.add_argument("--image_size", type=int, default=128, help="Size of the images. (128)")

# Number of channels in the training images. For color images this is 3
parser.add_argument("--nc", type=int, default=3, help="Number of channels in the images. (3)")

# Size of z latent vector (i.e. size of generator input)
parser.add_argument("--nz", type=int, default=256, help="Size of latent dimension (generator input). (256)")

# Size of feature maps in generator
parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator. (64)")

# Size of feature maps in discriminator
parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator. (64)")

# Number of training epochs
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs. (10)")

# Probability of experience replay
parser.add_argument("--er_prob", type=float, default=0., help="Probability to show discriminator past generations. (0)")

# Learning rate for optimizers
parser.add_argument("--lr_g", type=float, default=0.0005, help="Generator learning rate. (0.0005)")
parser.add_argument("--lr_d", type=float, default=0.0005, help="Discriminator learning rate. (0.0005)")
parser.add_argument("--lambda_mass", type=float, default=0., help="Weights of the Mass conservation law. (0)")
parser.add_argument("--lambda_pressure", type=float, default=0., help="Weights of the Pressure balance law. (0.)")
parser.add_argument("--lambda_PI", type=float, default=0., help="Weights of physics loss for the square dataset. (0.)")

# Beta1 hyperparameter for Adam optimizers
parser.add_argument("--beta_g", type=float, default=0.5, help="Generator's Adam beta. (0.5)")
parser.add_argument("--beta_d", type=float, default=0.5, help="Discriminator's Adam beta. (0.5)")

# Number of discriminator updates for each generator update
parser.add_argument("--n_critic", type=int, default=5, help="Number of discriminator updates for each generator update. Used in WGAN and WGAN-GP. (5)")

# Weight of gradient penalty
parser.add_argument("--lambda_gp", type=float, default=10., help="Weight of gradient penalty for WGAN-GP. (10)")

# Weight clip
parser.add_argument("--weight_clip", type=float, default=0.01, help="Clip weights of WGAN discriminator in (-weight_clip, weight_clip). (0.01)")

# Dropout
parser.add_argument("--dropout", type=float, default=0., help="Dropout for generator. (0)")

# Negative slope for LeakyReLU
parser.add_argument("--slope", type=float, default=0.01, help="Negative slope for LeakyReLU. (0.01)")

# Noise
parser.add_argument("--noise", type=float, default=0., help="Add noise ~gaussian(0,noise) to discriminator input, to promote pdf overlap. (0)")

# Number of GPUs available. Use 0 for CPU mode.
parser.add_argument('--single_gpu', action='store_true', help="Flag. If included uses a single GPU (gpu1)")
parser.add_argument("--gpu1", type=int, default=None, help="Index of the first GPU")
parser.add_argument("--gpu2", type=int, default=None, help="Index of the second GPU")

args = parser.parse_args()


#######################################
#           DATASET CREATION          #
#######################################


# Remove previous fake images per epoch
for file in glob.glob(os.path.join("[output_directory]", "*.png")):
    os.remove(file)

transform=transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.CenterCrop(args.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

paired_dataset = functions.PairedImageDataset(args.dataset_pressure, args.dataset_velocity, transform=transform)

# Determine number of classes
num_classes = len(paired_dataset.classes)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)

# Decide which device we want to run on
device = torch.device(f"cuda:{args.gpu1}" if torch.cuda.is_available() else "cpu")
print(device)


#########################################
#           FIXED PARAMETERS            #
#########################################


lambda_mass = args.lambda_mass
lambda_pressure = args.lambda_pressure
colormap_tensor = functions.custom_colormap(device)


##########################################################
#           NETWORK GEOMETRY AND INITIALIZATION          #
##########################################################

# Initialize models
generator = functions.UpsampleGenerator(args.nz, args.ngf, args.nc, num_classes, args.image_size, args.dropout, args.slope).to(device)
discriminator = functions.Discriminator(args.nc, args.ndf, num_classes, args.image_size, args.model, args.noise, args.dropout, args.slope).to(device)
# Setup optimizers
optimizerG = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta_g, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta_d, 0.999))

#########################################
#           TRAIN PREPARATION           #
#########################################


# Create fixed noise and label for monitoring progress
fixed_noise = torch.randn(1, args.nz, device=device)
fixed_label = torch.randint(0, num_classes, (1,), device=device)
# Initialize replay buffer
replay_buffer = []


####################################
#           TRAINING LOOP          #
####################################


# Record the start time
total_start_time = time.time()

# Lists to keep track of progress
G_loss = []
D_loss = []
W_loss = []
mass_loss_list = []
pressure_loss_list = []
img_list = []
iters = 0

if args.model == "GAN":
    criterion = nn.BCELoss()

    print("Starting Training Loop...")

    for epoch in range(args.num_epochs):
        for i, (data, labels) in enumerate(dataloader, 0):
            combined_image = data.to(device)
            labels = labels.to(device)

            b_size = combined_image.size(0)
            label_real = torch.full((b_size,), args.real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), args.fake_label, dtype=torch.float, device=device)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            discriminator.zero_grad()

            # Forward pass real batch through D with labels
            output_real = discriminator(combined_image, labels)
            errD_real = criterion(output_real.view(-1), label_real)
            errD_real.backward()

            # Generate fake data using Generator with labels
            noise = torch.randn(b_size, args.nz, device=device)
            fake_combined_image = generator(noise, labels)
            if args.er_prob > 0:
                fake_combined_image, labels, replay_buffer = functions.experience_replay(replay_buffer, fake_combined_image, labels, b_size, args.er_prob)

            # Forward pass fake data through D with labels
            output_fake = discriminator(fake_combined_image.detach(), labels)
            errD_fake = criterion(output_fake.view(-1), label_fake)
            errD_fake.backward()

            # Update Discriminator weights
            optimizerD.step()
            errD = errD_real.item() + errD_fake.item()
            D_loss.append(errD)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            output = discriminator(fake_combined_image, labels)
            errG = criterion(output.view(-1), label_real)

            # Optional physics loss
            if (lambda_mass or lambda_pressure> 0) and epoch >= 20:
                mass_loss, pressure_loss = ValveLoss.compute_losses(fake_combined_image, args.image_size, colormap_tensor)

                # Clamping the losses to avoid excessively high values
                mass_loss = torch.clamp(mass_loss, max=1.0)
                pressure_loss = torch.clamp(pressure_loss, max=1.0)
                if not torch.isfinite(mass_loss).all() or not torch.isfinite(pressure_loss).all():
                    print("Invalid loss values detected. Skipping this batch.")
                    mass_loss, pressure_loss = torch.tensor(0.0), torch.tensor(0.0)
                    errG *= 1.1

                total_G_loss = errG + lambda_mass * mass_loss + lambda_pressure * pressure_loss
                total_G_loss.backward()
                G_loss.append(total_G_loss.item())

                # Append the physics losses to their respective lists
                mass_loss_list.append(mass_loss.item())
                pressure_loss_list.append(pressure_loss.item())
            else:
                mass_loss, pressure_loss = torch.tensor(0.0), torch.tensor(0.0)
                errG.backward()
                G_loss.append(errG.item())

                # Append the physics losses to their respective lists
                mass_loss_list.append(mass_loss.item())
                pressure_loss_list.append(pressure_loss.item())

            # Update Generator weights
            optimizerG.step()

            # Save generated images periodically and at the last iteration
            if (iters % 500 == 0) or ((epoch == args.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_combined_image_fixed = generator(fixed_noise, fixed_label)
                    img_list.append(fake_combined_image_fixed.detach().cpu())

            # Print loss values every iteration
            if lambda_mass or lambda_pressure > 0:
                print(f'[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}] Loss D: {D_loss[-1]:.4f}, Loss G: {G_loss[-1]:.4f}, Loss mass conservation: {mass_loss:.4f}, Loss pressure: {pressure_loss:.4f}')
            else:
                print(f'[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}] Loss D: {D_loss[-1]:.4f}, Loss G: {G_loss[-1]:.4f}')
            iters += 1
    
        # Save the generated images, location to specify
        functions.save_separated_generated_images(generator, fixed_noise, device, ["output_dir"], epoch)

    # Save latest weights of the networks, location to specify
    torch.save(generator.state_dict(), ["output_dir"])
    torch.save(discriminator.state_dict(), ["output_dir"])
    
    # Save losses as arrays, location to specify
    G_loss = np.array(G_loss)
    D_loss = np.array(D_loss)
    mass_loss_array = np.array(mass_loss_list)
    pressure_loss_array = np.array(pressure_loss_list)
    np.save(["output_dir"], G_loss)
    np.save(["output_dir"], D_loss)
    np.save(["output_dir"], mass_loss_array)
    np.save(["output_dir"], pressure_loss_array)

###############################
###         WGAN-GP         ###
###############################

if args.model == "WGAN-GP":
    for epoch in range(args.num_epochs):
        for i, (data, labels) in enumerate(dataloader, 0):
            combined_image = data.to(device)
            labels = labels.to(device)

            b_size = combined_image.size(0)

            ############################
            # (1) Update D network: maximize D(x) - D(G(z))
            ###########################
            discriminator.zero_grad()

            # Train with all-real batch
            output_real = discriminator(combined_image, labels).view(-1)
            errD_real = output_real.mean()

            # Train with all-fake batch
            noise = torch.randn(b_size, args.nz, device=device)
            fake_combined_image = generator(noise, labels)
            if args.er_prob > 0:
                fake_combined_image, labels, replay_buffer = functions.experience_replay(replay_buffer, fake_combined_image, labels, b_size, args.er_prob)

            # Forward pass fake data through D with labels
            output_fake = discriminator(fake_combined_image.detach(), labels).view(-1)
            errD_fake = output_fake.mean()

            # Gradient penalty
            alpha = torch.rand(b_size, 1, 1, 1, device=device)
            interpolates = (alpha * combined_image + ((1 - alpha) * fake_combined_image)).requires_grad_(True)
            d_interpolates = discriminator(interpolates, labels).view(b_size, -1).mean(1)
            grad_outputs = torch.ones(b_size, device=device)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda_gp

            wasserstein_D = errD_real - errD_fake
            W_loss.append(wasserstein_D.item())
            errD = -wasserstein_D + gradient_penalty
            errD.backward(retain_graph=True)
            optimizerD.step()

            D_loss.append(errD.item())

            ############################
            # (2) Update G network: minimize -D(G(z))
            ###########################
            if i % args.n_critic == 0:
                generator.zero_grad()
                noise = torch.randn(b_size, args.nz, device=device)
                fake_combined_image = generator(noise, labels)
                output = discriminator(fake_combined_image, labels).view(-1)
                errG = -output.mean()

                # Initialize and compute baseline generator loss during the first 20 epochs
                if epoch == 0 and i == 0:
                    generator_loss_baseline = 0.0
                    count_baseline_samples = 0

                if epoch < 20:
                    generator_loss_baseline += errG.item()
                    count_baseline_samples += 1
                    if epoch == 19 and i == len(dataloader) - 1:
                        generator_loss_baseline /= count_baseline_samples
                        print(f'Baseline generator loss: {generator_loss_baseline:.4f}')

                # Adjust lambda values dynamically after 20th epoch
                if epoch >= 20:
                    scaling_factor = errG.item() / generator_loss_baseline
                    adjusted_lambda_mass = lambda_mass * scaling_factor
                    adjusted_lambda_pressure = lambda_pressure * scaling_factor
                else:
                    adjusted_lambda_mass = lambda_mass
                    adjusted_lambda_pressure = lambda_pressure

                # Apply custom losses after 20th epoch
                if (adjusted_lambda_mass or adjusted_lambda_pressure > 0) and epoch >= 20:
                    mass_loss, pressure_loss = ValveLoss.compute_losses(fake_combined_image, args.image_size, colormap_tensor)
                    
                    # Clamping the losses to avoid excessively high values
                    mass_loss = torch.clamp(mass_loss, max=1.0)
                    pressure_loss = torch.clamp(pressure_loss, max=1.0)
                    if not torch.isfinite(mass_loss).all() or not torch.isfinite(pressure_loss).all().all():
                        print("Invalid loss values detected. Skipping this batch.")
                        mass_loss, pressure_loss = torch.tensor(0.0), torch.tensor(0.0)
                        errG *= 1.1

                    total_G_loss = errG + adjusted_lambda_mass * mass_loss + adjusted_lambda_pressure * pressure_loss
                    total_G_loss.backward()
                    G_loss.append(total_G_loss.item())

                    # Append the physics losses to their respective lists
                    mass_loss_list.append(mass_loss.item())
                    pressure_loss_list.append(pressure_loss.item())
                else:
                    mass_loss, pressure_loss = torch.tensor(0.0), torch.tensor(0.0)
                    errG.backward()
                    G_loss.append(errG.item())
                
                    # Append the physics losses to their respective lists
                    mass_loss_list.append(mass_loss.item())
                    pressure_loss_list.append(pressure_loss.item())

                optimizerG.step()

            # Save generated images periodically and at the last iteration
            if (iters % 500 == 0) or ((epoch == args.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_combined_image_fixed = generator(fixed_noise, fixed_label)
                    img_list.append(fake_combined_image_fixed.detach().cpu())

            # Print loss values every iteration
            if lambda_mass or lambda_pressure > 0:
                print(f'[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}] Loss D: {D_loss[-1]:.4f}, Loss G: {G_loss[-1]:.4f}, Wasserstein D: {wasserstein_D:.4f}, Loss mass conservation: {mass_loss:.4f}, Loss pressure: {pressure_loss:.4f}')
            else:
                print(f'[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}] Loss D: {D_loss[-1]:.4f}, Loss G: {G_loss[-1]:.4f}, Wasserstein D: {wasserstein_D:.4f}')
            iters += 1

        # Save the generated images, location to specify
        functions.save_separated_generated_images(generator, fixed_noise, device, b_size, epoch)
    
    # Save latest weights of the networks, location to specify
    torch.save(generator.state_dict(), ["output_dir"])
    torch.save(discriminator.state_dict(), ["output_dir"])

    # Save losses as arrays, location to specify
    G_loss = np.array(G_loss)
    D_loss = np.array(D_loss)
    W_loss = np.array(W_loss)
    mass_loss_array = np.array(mass_loss_list)
    pressure_loss_array = np.array(pressure_loss_list)

    np.save(["output_dir"], G_loss)
    np.save(["output_dir"], D_loss)
    np.save(["output_dir"], W_loss)
    np.save(["output_dir"], mass_loss_array)
    np.save(["output_dir"], pressure_loss_array)

##########################################
#           POST TRAINING CHECK          #
##########################################

# Save and plot losses, location to specify
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_loss, label="Generator Loss")
plt.plot(D_loss, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(["output_dir"])
plt.close()

# Record the end time
total_end_time = time.time()

# Calculate the total time taken
total_time = total_end_time - total_start_time

# Convert the total time to hours, minutes, and seconds
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

# Print the total time taken
print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")