from Unet import UNet
from Test import *
import pandas as pd
model = UNet()
path_to_test_data = "tim's_data/Test"
path_to_labels = "tim's_data/labels.csv"
path_to_weights = '3.1/weights'
thr = 20
size = 25
# 1 for clot 0 for healthy
labels = pd.read_csv(path_to_labels, names=['ID', 'condition'], dtype={'condition': 'Int64'})


def main():
    # compare the generated and original image, calculate the difference score between them and predict label and accuracy
    model.load(f'expiriment report/{path_to_weights}/encoder', f'expiriment report/{path_to_weights}/decoder')
    get_predictions_and_accuracy(model, path_to_test_data, labels, thr, size)


if __name__ == '__main__':
    main()
