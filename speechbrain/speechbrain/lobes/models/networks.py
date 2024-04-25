import torch
import torch.nn as nn
import numpy as np


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], num_smp=2):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.num_smp = num_smp
        conv = nn.Sequential(*[nn.Conv2d(1, 1, 3, 1), nn.ReLU(), nn.Conv2d(1, 1, 3, 1)])
        mlp = nn.Sequential(*[nn.Linear(1, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        if len(self.gpu_ids) > 0:
            conv.cuda()
            mlp.cuda()
        setattr(self, 'conv', conv)
        setattr(self, 'mlp', mlp)

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        for feat_id, feat in enumerate(feats):
            if self.use_mlp:
                conv = getattr(self, 'conv')
                feat = feat.permute(1, 0, 2, 3)
                feat = conv(feat)
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp')
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
