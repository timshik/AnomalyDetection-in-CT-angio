from Unet import UNet
from DeepPrior import DeepPrior
from WAE import Wae
from MainProcess import *
from torch import nn
from Lookahead import *
from Test import *

device = get_device()
batch_size = 8
epochs = 1500
path_to_test_data = "tim's_data/Test"
path_to_data = "tim's_data/Train"
path_to_weights = 'weights'  #########3
train_test_rate = 20
lr = 1e-2

# model
model = Wae()
# loss
loss = nn.MSELoss(reduction='sum')  # nn.L1Loss(), reduction='sum'
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.8, 0.980), eps=1e-08)  # the parameters are default for now
# optimizer = Lookahead(optimizer)  # doesn't work with gpu for now
mask = True
mask_rate = 0.0001


def main():
    main_process(model, path_to_data, loss, optimizer, epochs, path_to_weights, batch_size, train_test_rate, mask, mask_rate)
    # model.load(f'expiriment report/{path_to_weights}/encoder', f'expiriment report/{path_to_weights}/decoder')
    # test(model, path_to_test_data, True, 0.0001)  # no mask for now


if __name__ == '__main__':
    main()
