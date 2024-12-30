from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PATH = '/content/drive/My Drive/Datasets/BSD100'

# Defining custom dataset class
class BSD100DataSet(Dataset):
    def __init__(self, dataset_path = DATASET_PATH, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images_list = []

    def __len__(self):
        return len(os.listdir(self.dataset_path)) // 2

    def __getitem__(self, idx):
        os.chdir(self.dataset_path)
        if isinstance(idx, slice):
            return [self.__getitem__(ii) for ii in range(*idx.indices(len(self)))] # see PyTorch notes 'slice type' for explanation
        elif isinstance(idx, int):
            img_path = os.path.join(self.dataset_path, self.name_constructor(idx))
            image = read_image(img_path)

            if self.transform:
                image = self.transform(image)

            input = image.clone() + torch.randn(image.shape)*0.1
            target = image.clone()
            input = input.to(device)
            target = target.to(device)

            return input, target

    def name_constructor(self, idx):
        leading_zeros = 3 - len(str(idx+1))
        ret_str = 'img_' + leading_zeros * '0' + str(idx+1) + '_SRF_2_HR.png'
        return ret_str

# defining separate classes for train and test set as slicing leads to deterministic one time transforms
# this ensures that the __getitem__ method of BSD100DataSet class is always called while accessing the train and test set
# it is sufficient to define the __len__ method only and inherit everything else ( for dataloader to know where to stop )
# the dataloader internally calls the __getitem__ method of the dataset class instance passed to it everytime it accesses data
class TrainSet(BSD100DataSet):
    def __init__(self, dataset_path = DATASET_PATH, transform=None):
        super().__init__(dataset_path, transform)
        self.size = 90
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        if( idx >= self.size):
            raise IndexError
        else:
            return super().__getitem__(idx)

class TestSet(BSD100DataSet):
    def __init__(self, dataset_path = DATASET_PATH, transform=None):
        super().__init__(dataset_path, transform)
        self.size = 10
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        if( idx >= self.size):
            raise IndexError
        else:
            return super().__getitem__(idx + 90)

