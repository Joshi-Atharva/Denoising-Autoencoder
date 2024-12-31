# defining dataset class for the FFHQ dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PATH = 'C:/Users/athar/MLprojects/dataloader_task/Datasets/FFHQ'
TRAIN_SIZE = 256
TEST_SIZE = 32
NOISE_FACTOR = 0.1

class FFHQDataSet(Dataset):
    def __init__(self, dataset_path = DATASET_PATH, transform = None, noise_factor = NOISE_FACTOR):
        self.dataset_path = dataset_path
        self.transform = transform
        self.noise_factor = noise_factor
        self.len = 288
        # self.len = len(os.listdir(dataset_path))
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        if(isinstance(idx, slice)):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        else:
            name = self.name_constructor(idx)
            img_path = os.path.join(self.dataset_path, name)
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            
            input = image.clone() + torch.randn(image.shape)*self.noise_factor
            target = image.clone()
            input = input.to(device)
            target = target.to(device)

            return input, target
    def name_constructor(self, idx):
        leading_zeros = 5 - len(str(idx))
        name = '0' * leading_zeros + str(idx) + '.png'
        return name

class TrainSet(FFHQDataSet):
    def __init__(self, dataset_path = DATASET_PATH, transform = None, size = TRAIN_SIZE):
        super().__init__(dataset_path, transform) # super() is used to refer to parent class. That way this line is calling the __init__ method of the parent class with specified parameters
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return super().__getitem__(idx)

class TestSet(FFHQDataSet):
    def __init__(self, dataset_path = DATASET_PATH, transform = None, size = TEST_SIZE):
        super().__init__(dataset_path, transform)
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return super().__getitem__(idx)
