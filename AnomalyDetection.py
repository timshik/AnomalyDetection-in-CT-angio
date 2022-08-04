from Unet import UNet
from MainProcess import *
from torch import nn
from Lookahead import *
from Test import test

device = get_device()
batch_size = 8
epochs = 500
path_to_data = "tim's_data/Train"
path_to_test_data = "tim's_data/Test"
path_to_weights = 'weights'
train_test_rate = 0.3
lr = 1e-2

# todo try to leave the dimensions as they are (1024X1024) (need GPU)
# todo enlarge the mask rate
# todo try to reconstruct normal and distal clots and see the error of reconstruction
# if the reconstruction will be perfect we can try to reconstruct distal clot scan and then subtract it from the original scan, the clot shouldnt be in the reconstructed image.

# model
model = UNet()
# loss
loss = nn.MSELoss(reduction='sum')  # nn.L1Loss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.8, 0.980), eps=1e-08)  # the parameters are default for now
# optimizer = Lookahead(optimizer)  # doesnt work with gpu for now
mask = True
mask_rate = 0.0001


def main():
    # main_process(model, path_to_data, loss, optimizer, epochs, path_to_weights, batch_size, train_test_rate, mask, mask_rate)
    # test
    model.load(f'{path_to_weights}/encoder', f'{path_to_weights}/decoder')
    test(model, path_to_test_data, True, 0.0001)  # no mask for now


if __name__ == '__main__':
    main()

