import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, embedding_dim):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        # Increase the number of features in the first layer
        self.fc = nn.Linear(input_dim + embedding_dim, 1024 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            # Add an additional upsampling block
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # First upsampling block
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Second upsampling block
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Third upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer
            nn.ConvTranspose2d(64, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, class_labels):
        # Ensure class_labels is a tensor of long integers
        if not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=noise.device)
        elif class_labels.dtype != torch.long:
            class_labels = class_labels.long()

        # Embed the class labels
        embedded_labels = self.embedding(class_labels).view(-1, self.embedding.embedding_dim)

        # Concatenate noise and embedded labels
        combined_input = torch.cat((noise, embedded_labels), dim=1)

        # Forward pass through the fully connected layer
        x = self.fc(combined_input)
        x = x.view(-1, 1024, 4, 4)  # Adjust the number of features to match the fc layer

        # Forward pass through the convolutional layers
        x = self.conv_blocks(x)

        # Resize images to the desired output size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # Rescale output to the range [0, 1] for visualization
        x = (x + 1) / 2
        return x
