import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


def transform_img(image, thr):
    # image[image > thr] = 255
    # image[image <= thr] = 0
    return image


def data_prep(path):
    scans_path = os.listdir(path)
    scans = []
    for scan in sorted(scans_path):
        path_to_scan = f'{path}/{scan}/SAG'
        try:
            path_to_scan += f'/{os.listdir(path_to_scan)[0]}'
            image = cv2.imread(path_to_scan, 0)
            image = transform_img(image, 130)  # not sure if needed
            # image = cv2.resize(image, (128, 128))  #.astype(float)/255.0-0.5
            scans.append([scan, image])
        except:
            continue

    return np.asarray(scans)


class GroupData:
    def __init__(self, path):
        self.data = data_prep(path)
        self.train = None
        self.val = None

    def my_train_test_split(self, split_rate, constant_split=None):
        self.train, self.val = train_test_split(self.data, test_size=split_rate)

    def get_train_val_datasets(self):
        return self.train, self.val

    def get_dataset(self):
        return self.data


# we first subtract the image from 255 because we want the mask be in shape of a white squares
def create_mask(img, rate):
    img_reversed = 255 - img
    size = len(img_reversed) * len(img_reversed[0])
    array = np.ones(size)
    indices = np.random.choice(np.arange(size), int(size * rate), replace=False)
    array[indices] = 0
    mask = np.reshape(array, (len(img_reversed), len(img_reversed[0])))
    return mask


def mask_image_pixels(img, rate):
    mask = create_mask(img, rate)
    return 255 - (255-img) * mask.astype(int)  # we want the mask to be white


def mask_image_batches(img, rate, size=25):
    mask = create_mask(img, rate)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i, j] == 0:
                mask[i:i+size, j:j+size] = 2
    mask[mask == 2] = 0
    return 255 - (255-img) * mask.astype(int)


class TwoGroupDataset(Dataset):
    def __init__(self, data, mask, rate):
        self.data = data
        self.mask = mask
        self.rate = rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        subject = self.data[item][0]
        data = self.data[item][1]
        orig_image = data.reshape(1, data.shape[0], data.shape[1])
        if self.mask:
            masked_image = mask_image_batches(data, self.rate)
            masked_image = masked_image.reshape(1, masked_image.shape[0], masked_image.shape[1])
        else:
            masked_image = orig_image
        return subject, orig_image, masked_image


def get_train_test_loaders(path, split_rate, batch_size, mask, rate):
    my_data = GroupData(path)
    my_data.my_train_test_split(split_rate)
    train, val = my_data.get_train_val_datasets()
    train_set = TwoGroupDataset(train, mask, rate)
    val_set = TwoGroupDataset(val, mask, rate)
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_data_loader, val_data_loader


def get_all_data_dataloader(path,  mask, rate):
    my_data = GroupData(path)
    data = my_data.get_dataset()
    data_set = TwoGroupDataset(data, mask, rate)
    data_loader = DataLoader(data_set, shuffle=True)
    return data_loader
