import torch
from torch import nn
from Barbershop.models.Net import Net
from Barbershop.utils.bicubic import BicubicDownSample
from Barbershop.models.face_parsing.model import BiSeNet, seg_mean, seg_std

class Tool(nn.Module):
    def __init__(self, opts):
        super(Tool, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)
        self.load_segmentation_network()

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()