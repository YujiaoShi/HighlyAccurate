import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128*2, kernel_size=(3, 3), padding=(1, 1)):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class CoordEncoder(nn.Module):
    def __init__(self, cfg, hidden_dim=128, num_layers=4):
        super(CoordEncoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, hidden_dim, kernel_size=(1, 1), padding=0),
        ])
        # (2 * cfg.models.num_encoding_fn_xyz + 1)*2
        for idx in range(num_layers):
            self.convs.extend([
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), padding=0),
            ])

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)

        return x


class PoseFeature(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim=128, num_layers=4):
        super(PoseFeature, self).__init__()
        self.CoordEncoder = CoordEncoder(cfg, hidden_dim, num_layers)
        # self.CoordEncoder = self.construct_layers(hidden_dim, num_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, feat, coord):
        x = self.conv1(feat)
        y = self.CoordEncoder(coord)
        z = torch.cat([x, y], dim=1)
        z = self.conv2(z)
        return z


class GRUPoseRefine(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim=128, num_layers=4):
        super(GRUPoseRefine, self).__init__()

        self.PoseFeature = PoseFeature(cfg, input_dim, hidden_dim, num_layers)
        self.PoseHeader = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim//2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim//2), 4),
            nn.Tanh()
        )
        self.convGRU = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)

    def forward(self, query_feat, pred_feat, pred_grids, h):
        pred_grids = F.interpolate(pred_grids.permute(0, 3, 1, 2), size=pred_feat.shape[2:], mode='bilinear')

        poseFeat = self.PoseFeature(pred_feat, pred_grids)
        x = torch.cat([query_feat, poseFeat], dim=1)
        h = self.convGRU(h, x)

        h_ = torch.mean(h, dim=[-1, -2])
        delta_pose = self.PoseHeader(h_)

        return h, delta_pose


class NNrefine(nn.Module):
    def __init__(self):
        super(NNrefine, self).__init__()
        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear3 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(64, 16),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(16, 3),
                                     nn.Tanh())

    def forward(self, pred_feat, ref_feat):
        r = pred_feat - ref_feat  # [B, C, H, W]
        B, C, _, _ = r.shape
        if C == 256:
            x = self.linear0(r)
        elif C == 128:
            x = self.linear1(r)
        elif C == 64:
            x = self.linear2(r)
        elif C == 16:
            x = self.linear3(r)

        x = torch.mean(x, dim=[2, 3])

        y = self.mapping(x)  # [B, 3]
        return y
