import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SketchDataset
from model import Generator

# Define hyperparameters
batch_size = 32
num_epochs = 100
learning_rate = 0.0002
input_dim = 100  # Size of noise vector
output_dim = 1  # Number of channels in output image
num_classes = 6  # Number of object categories
embedding_dim = 50  # Dimensionality of class embeddings

# Define transformations for image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std if needed
])

# Define paths
dataset_root = r'C:\Users\This PC\Desktop\nihal project copy\png'  # Update with the correct path
generated_images_dir = 'generated_images'  # Directory to save generated images

# Create dataset and data loader
train_dataset = SketchDataset(dataset_root, transform=image_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the generator model with the updated architecture
generator = Generator(input_dim, output_dim, num_classes, embedding_dim * 2)  # Adjusted embedding dimension

# Define the optimizer and loss function
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Use Mean Squared Error Loss for image generation tasks

# Check if the directory to save generated images exists
if not os.path.exists(generated_images_dir):
    os.makedirs(generated_images_dir)
    print("Generated images directory created successfully.")

losses = []

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Generate random noise
        noise = torch.randn(labels.size(0), input_dim)

        # Forward pass
        generated_images = generator(noise, labels)

        # Compute the loss
        loss = criterion(generated_images, images)  # Compare generated images with real images

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

        # Save generated images for inspection
        if (batch_idx + 1) % 50 == 0:
            for i, image in enumerate(generated_images):
                # Rescale the images to the range [0, 1]
                img = (image + 1) * 0.5
                img = transforms.ToPILImage()(img)
                save_path = os.path.join(generated_images_dir, f"generated_image_epoch{epoch}_batch{batch_idx}_{i}.png")
                img.save(save_path)

# Save the trained model
torch.save(generator.state_dict(), 'generator.pth')

# Save the loss values to a text file
with open('loss_values.txt', 'w') as file:
    for loss_value in losses:
        file.write(f"{loss_value}\n")
