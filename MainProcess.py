import torch
import cv2
from DataLoader import get_train_test_loaders
import cv2
import os
Is_GPU_PC = False
from Utils import *

device = get_device()


def save_batch(dir, orig, noisy, created, subject):
    noisy = noisy.detach().numpy()
    created = created.detach().numpy()
    orig = orig.detach().numpy()
    for j in range(len(created)):
        makedir(f'{dir}/{subject[j]}')
        cv2.imwrite(f'{dir}/{subject[j]}/orig_noisy.png', un_normalize_image(noisy[j][0]))
        cv2.imwrite(f'{dir}/{subject[j]}/orig.png', un_normalize_image(orig[j][0]))
        cv2.imwrite(f'{dir}/{subject[j]}/created.png', un_normalize_image(created[j][0]))


def train(model, train_loader, val_loader, loss, optimizer, epochs):
    tr_losses = []
    dev_losses = []
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            running_loss = 0
            subject, img, img_noisy = data
            # putting the data into the GPU CUDA device
            img_noisy = img_noisy.to(device)
            img = img.to(device)
            # prepare the optimizer
            optimizer.zero_grad()
            # defining the type of data as float, otherwise it doesn't work
            model = model.float()
            outputs = model(img_noisy.float())
            tr_loss = loss(outputs, img.float())
            tr_loss.backward()
            optimizer.step()
            running_loss += tr_loss.item()
            if batch_idx % 1 == 0:  # printing the loss every epoch
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, tr_loss.item()))
            if epoch % 20 == 0 and batch_idx % 9 == 0:
                makedir(f'train_{epoch}')
                dir = f'train_{epoch}'
                save_batch(dir, img, img_noisy, outputs, subject)

        model.eval()
        with torch.no_grad():
            runing_dev_loss = 0
            for data in val_loader:
                subject, img, img_noisy = data
                model = model.float()
                outputs = model(img_noisy.float())
                dev_loss = loss(outputs, img.float())
                runing_dev_loss += dev_loss
                if epoch % 20 == 0:
                    makedir(f'val_{epoch}')
                    dir = f'val_{epoch}'
                    save_batch(dir, img, img_noisy, outputs, subject)
            tr_losses.append(running_loss / len(train_loader))
            dev_losses.append(runing_dev_loss / len(val_loader))

    return tr_losses, dev_losses


def evaluate(model, loader, train_or_test):
    makedir(train_or_test)
    model.eval()
    for data in loader:
        subject, img, img_noisy = data
        model = model.float()
        out = model(img_noisy.float())
        dir = f'{train_or_test}'
        save_batch(dir, img, img_noisy, out, subject)


def main_process(model, path_to_data, loss, optimizer, epochs, path_to_weights, batch_size, train_test_rate=10):
    train_loader, val_loader = get_train_test_loaders(path_to_data, train_test_rate, batch_size)
    tr_losses, val_losses, = train(model, train_loader, val_loader, loss, optimizer, epochs)
    output_learning_graph(tr_losses, val_losses)
    makedir(path_to_weights)
    makedir(f'{path_to_weights}')
    model.save(f'{path_to_weights}/encoder', f'{path_to_weights}/decoder')
    evaluate(model, train_loader, 'train')
    print('------------------------------------------')
    evaluate(model, val_loader, 'test')


