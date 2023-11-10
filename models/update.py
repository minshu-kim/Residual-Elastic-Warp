import torch
import torch.nn as nn
from .utils import *

class SepConvGRU_kp(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=162, num_kp = 8):
        super(SepConvGRU_kp, self).__init__()
        self.convz1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        lst = []
        max_pool = num_kp//4 
        if max_pool == 3:
            self.enc = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  ##32
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),   ##16

                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),  ##14
                nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),   ##12
                nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                nn.ReLU(), 
            )
        elif max_pool == 4:
            for i in range(2):
                layer = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                    nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                lst.append(layer)
            self.enc = nn.ModuleList(lst)
        else :
            for i in range(5-max_pool):
                layer = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                    nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                lst.append(layer)
            self.enc = nn.ModuleList(lst)

        self.head = nn.Sequential(
                nn.Conv2d(hidden_dim, 2, 1, stride=1)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        head = h

        for layer in self.enc:
            head = layer(head)
        delta = self.head(head).flatten(1)
        return h, delta

class SepConvGRU_kp_ihn(nn.Module):
    def __init__(self, sz=32, hidden_dim=96, input_dim=162):
        super(SepConvGRU_kp_ihn, self).__init__()
        self.convz1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))

        if sz==32:
            self.enc = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif sz==64:
            self.enc = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.head = nn.Sequential(
                nn.Conv2d(hidden_dim, 2, 1, stride=1)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        head = h
        head = self.enc(head)
        delta = self.head(head)
        return h, delta

class SepConvGRU2(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=290):
        super(SepConvGRU2, self).__init__()
        self.convz1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(input_dim+hidden_dim, hidden_dim, (5,1), padding=(2,0))

        self.neck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head1 = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, 3, padding=1, stride=1)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, 2, 3, padding=1, stride=1)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        n = self.neck(h)
        delta_64 = self.head1(n).flatten(1)
        delta_32 = self.head2(n)
        return h, delta_64, delta_32
 

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=162):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, 2, 3, padding=1, stride=1),
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
       
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        delta = self.head(h).flatten(1)
        return h, delta


class GRU(nn.Module):
    def __init__(self, sz, num_kp=9):
        super().__init__()
        # self.gru = SepConvGRU(input_dim=290, hidden_dim=128)
#        self.gru = SepConvGRU(input_dim=292, hidden_dim=128)
        self.gru = ConvGRU(input_dim = 164, hidden_dim = 128)
    def forward(self, corr, diff, state):

        x = torch.cat([corr, diff], dim=1)
        state, delta_flow = self.gru(state, x)

        return state, delta_flow


class GRU_kp(nn.Module):
    def __init__(self, sz, num_kp=9, input_dim=164, hidden_dim=96):
        super().__init__()
        self.gru = SepConvGRU_kp(hidden_dim, input_dim, num_kp)
    def forward(self, motion_feature, inp, state):
        x = torch.cat([motion_feature, inp], dim=1)
        state, delta_flow = self.gru(state, x)

        return state, delta_flow

class GRU_kp_ihn(nn.Module):
    def __init__(self, sz, input_dim=164, hidden_dim=96):
        super().__init__()
        self.gru = SepConvGRU_kp_ihn(sz, hidden_dim, input_dim)
    def forward(self, motion_feature, inp, state):
        x = torch.cat([motion_feature, inp], dim=1)
        state, delta_flow = self.gru(state, x)

        return state, delta_flow


class kp_tcell(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=96):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ##32
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   ##16

            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),  ##14
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1),   ##12
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, corr, flow):
        x = torch.cat([corr, flow], dim=1)
        delta = self.head(self.enc(x))

        return delta

class kp_tcell_dirichlet(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=96, thres=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ##32
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   ##16
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1), #14
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1), #12
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1), #10
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
        )
        self.threshold = thres
        self.head = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, corr, flow):
        x = torch.cat([corr, flow], dim=1)
        delta = self.head(self.enc(x))
        delta = torch.tanh(delta) * self.threshold

        return delta

class kp_hcell(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=96):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.head = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, corr, flow):
        x = torch.cat([corr, flow], dim=1)
        delta = self.head(self.enc(x))

        return delta

class kp_flow(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=96):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=hidden_dim//8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, 1),
            nn.Tanh()
        )

    def forward(self, corr, flow):
        x = torch.cat([corr, flow], dim=1)
        delta = self.head(self.enc(x))

        return delta
