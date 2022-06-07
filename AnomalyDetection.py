from DataLoader import get_train_test_loaders
import torch

batch_size = 16
epochs = 300
path_to_data = "tim's_data"
path_to_weights = 'weights'
train_test_rate = 10
lr = 1e-2

# model
# model = Wae()
# loss
# loss = nn.MSELoss()  # nn.L1Loss()  # nn.MSELoss()
# optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.8, 0.980), eps=1e-08)  # the parameters are default for now
# optimizer = Lookahead(optimizer)

# if train = 1 we train AE( group 0 - first AE, group 1 - second AE)
# else we evaluate on final AE


def main():
    train_loader, test_loader = get_train_test_loaders(path_to_data, train_test_rate, batch_size)


if __name__ == '__main__':
    main()
