import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class SketchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_descriptions = []

        # Mapping from description to index
        self.description_to_index = {
            'alarmclock': 0,
            'apple': 1,
            'bicycle': 2,
            'calculator': 3,
            'cigarette': 4,
            'cow': 5
        }

        # Read the filelist.txt file
        with open(os.path.join(root_dir, 'filelist.txt'), 'r') as file:
            for line in file:
                line = line.strip()
                image_folder, image_file = line.split('/')  # Split the line into folder and file
                image_path = os.path.join(root_dir, image_folder, image_file)  # Full path to the image
                description = image_folder  # Use the folder name as the description
                self.image_descriptions.append((image_path, self.description_to_index[description]))

    def __len__(self):
        return len(self.image_descriptions)

    def __getitem__(self, idx):
        img_path, label_index = self.image_descriptions[idx]
        sketch_img = Image.open(img_path).convert('L')  # Convert to grayscale

        # Apply the transformation if it exists
        if self.transform:
            sketch_img = self.transform(sketch_img)

        label_tensor = torch.tensor(label_index, dtype=torch.long)

        return sketch_img, label_tensor
