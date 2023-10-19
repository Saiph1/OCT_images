# modified from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
# import os
import glob
from PIL import Image
from torchvision import datasets, transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files_A = sorted(glob.glob(root_dir+"/test_A/*.png"))
        dup = glob.glob(root_dir+"/data_B/000010.oct.png")
        self.image_files_B = [item for item in dup for _ in range(len(self.image_files_A))]
        # self.image_files_B = sorted(glob.glob(root_dir+"/data_B/*.png"))

    def __len__(self):
        return len(self.image_files_B)

    def __getitem__(self, index):
        image_path = self.image_files_A[index]
        # target_path = self.image_files_B[index]
        target_path = self.image_files_B[0]
        image = Image.open(image_path)
        target = Image.open(target_path)

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)
        
        return image, target


# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([512,3200]),
    transforms.Normalize((0.5,), (0.5,)) # Assume that the data is normal distribution, with mean 0.5 and SD 0.5. 
    # Here the z score normalization is carried out such taht the mean becomes 0 and SD = 1. 
    # This will allow better convergence and training speed. 
])

# Create an instance of the custom dataset
custom_dataset = CustomDataset(root_dir='./data', transform=transform)

# Create a data loader for the dataset
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=4, shuffle=True, num_workers=4)