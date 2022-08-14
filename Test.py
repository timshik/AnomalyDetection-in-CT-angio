import copy

import torch
from Utils import *
from MainProcess import evaluate
from DataLoader import *
import cv2
import matplotlib.pyplot as plt

device = get_device()


def save_batch(dir, orig, noisy, created):

    makedir(f'{dir}')
    cv2.imwrite(f'{dir}/orig_noisy.png', un_normalize_image(noisy))
    cv2.imwrite(f'{dir}/orig.png', un_normalize_image(orig))
    cv2.imwrite(f'{dir}/created.png', un_normalize_image(created))
    cv2.imwrite(f'{dir}/subtracted.png', np.abs(un_normalize_image(created)-un_normalize_image(orig)))


def test(model, path, mask, rate):
    data = get_all_data_dataloader(path, mask, rate)
    evaluate(model, data, 'test')


# in the generation task we created random mask with multiply squares, here we are creating mask with one square over a specific indices
def mask_image_one_square(img, i, j, h_kernel, w_kernel):
    masked_image = copy.deepcopy(img)
    masked_image[i: i+h_kernel, j: j+w_kernel] = 255
    return masked_image


def calc_score(model, img, kernel_size, stride, subject):
    model.eval()
    model = model.to(device)
    model = model.float()
    total_generated = np.zeros((1024, 1024)) ####
    if isinstance(kernel_size, int):
        h_kernel = kernel_size
        w_kernel = kernel_size
    else:
        h_kernel = kernel_size[0]
        w_kernel = kernel_size[1]
    diff = []

    i = 0
    while i + h_kernel <= len(img):
        j = 0
        while j + w_kernel <= len(img[0]):
            masked = torch.Tensor(mask_image_one_square(img, i, j, h_kernel, w_kernel))
            masked = torch.unsqueeze(masked.reshape(1, masked.shape[0], masked.shape[1]), dim=0)
            masked = masked.to(device)
            generated = model(masked.float()).cpu().detach().numpy().squeeze()
            total_generated[i:i + h_kernel, j:j + w_kernel] = generated[i:i + h_kernel, j:j + w_kernel] ###
            original_square_mean = np.mean(img[i:i + h_kernel, j:j + w_kernel])
            generated_square_mean = np.mean(generated[i:i + h_kernel, j: j + w_kernel])
            diff.append(([i, j], (abs(original_square_mean - generated_square_mean))))  # /(225 - original_square_mean))))

            # save_batch(f'{i},{j}', img, masked.cpu().detach().numpy().squeeze(), generated)
            j += stride

        i += stride

    cv2.imwrite(f'{subject}total_generated.png', total_generated)

    return diff, max(np.array(diff)[:, 1])


def get_predictions_and_accuracy(model, path, labels, thr=20, size=50, stride=50):
    scores = []
    makedir('diffs_heatmaps')
    ordered_labels = []
    for subject in os.listdir(path):
        ordered_labels.append(labels[labels['ID'] == int(subject)]['condition'].to_numpy()[0])
        path_to_scan = f'{path}/{subject}/SAG'
        img = cv2.imread(f'{path_to_scan}/{os.listdir(path_to_scan)[0]}', 0)
        # img = quantize_array(img)
        img = transform_img(img, 160)
        diff, score = calc_score(model, img, size, stride, subject)
        # draw heat map of the differences
        diff_to_img(img, diff, size, f'diffs_heatmaps/{subject}.png')
        predicted = int(score > thr)
        scores.append([subject, score, predicted])
        print(scores[-1])
    scores = np.array(scores).astype(float)
    create_graph(ordered_labels, scores[:, 1])
    accuracy = sum(np.array(ordered_labels) == scores[:, 2]) / len(ordered_labels)*100
    print(accuracy)



