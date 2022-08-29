import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
import os
from PIL import Image
import PIL
Is_GPU_PC = True


def transform_img(image, thr):
    image[image > thr] = 255
    image[image <= thr] = 0
    return image


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
    tr_losses_norm = [x if x < 1500000000 else 1500000000 for x in tr_losses]
    dev_losses_norm = [x if x < 1500000000 else 1500000000 for x in dev_losses]
    iteration = np.arange(0, len(tr_losses_norm))
    plt.plot(iteration, tr_losses_norm, 'g-', iteration, dev_losses_norm, 'r-')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    green_patch = mpatches.Patch(color='green', label='train')
    red_patch = mpatches.Patch(color='red', label='test')
    plt.legend(handles=[green_patch, red_patch])
    plt.savefig('graph.png')
    plt.show()


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N+4)
    return mycmap


def diff_to_img(img, diff, kernel_size, path):
    diff_img = np.zeros(img.shape)
    for index, value in diff:
        i, j = index
        diff_img[i:i+kernel_size, j:j+kernel_size] = value
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    cb = ax.contourf(diff_img, cmap=transparent_cmap(plt.cm.Reds))
    plt.colorbar(cb)
    plt.savefig(path)


def create_graph(y, x):
    # for better visualization
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('score')
    plt.ylabel('condition')
    plt.savefig(f'score_graph.png')


def quantize_array(arr):
    img2 = Image.fromarray(arr)
    img_qa = img2.quantize(7)
    arr_qa = np.asarray(img_qa)
    for key in img_qa.palette.colors.keys():
        arr_qa = np.where(arr_qa != img_qa.palette.colors[key], arr_qa,  key[0])  ### ???

    return arr_qa
