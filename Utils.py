import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
import os
Is_GPU_PC = False


def get_device():
    if Is_GPU_PC:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    return device


def un_normalize_image(img):
    return img  # np.clip(img + 0.5, 0, 1)*255


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_decoded_image(img, name):
    save_image(img.view(img.shape), name)


def output_learning_graph(tr_losses, dev_losses):
    # for better visualization
    tr_losses_norm = [x if x < 7000 else 7000 for x in tr_losses]
    dev_losses_norm = [x if x < 7000 else 7000 for x in dev_losses]
    iteration = np.arange(0, len(tr_losses_norm))
    plt.plot(iteration, tr_losses_norm, 'g-', iteration, dev_losses_norm, 'r-')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    green_patch = mpatches.Patch(color='green', label='train')
    red_patch = mpatches.Patch(color='red', label='test')
    plt.legend(handles=[green_patch, red_patch])
    plt.savefig('graph.png')
    plt.show()