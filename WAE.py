import torch
import torch.nn as nn
import torch.nn.functional as F

drop_rate = 0.0


class Encoder(nn.Module):
    def __init__(self, n_channel, dim_h, n_z):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        # self.fc = nn.Linear(self.dim_h * 8 * 8 * 8, self.n_z)

    def forward(self, x):
        x = self.main(x)
        # x = x.view(-1, self.dim_h * 8 * 8 * 8)
        # x = self.fc(x)
        return x

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


class Decoder(nn.Module):
    def __init__(self, n_channel, dim_h, n_z):
        super(Decoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 8 * 8),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.ConvTranspose2d(self.dim_h * 2, dim_h, 4, 2, 1),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.ConvTranspose2d(self.dim_h, n_channel, 4, 2, 1),

        )

    def forward(self, x):
        # x = self.proj(x)
        # x = x.view(-1, self.dim_h * 8, 8, 8)
        x = self.main(x)
        return x

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


class Wae(nn.Module):
    def __init__(self, n_channel=1, dim_h=16, n_z=2**11):
        super(Wae, self).__init__()
        self.encoder = Encoder(n_channel, dim_h, n_z)
        self.decoder = Decoder(n_channel, dim_h, n_z)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def save(self, path_enc, path_dec):
        self.encoder.save(path_enc)
        self.decoder.save(path_dec)

    def load(self, path_enc, path_dec):
        self.encoder.load(path_enc)
        self.decoder.load(path_dec)