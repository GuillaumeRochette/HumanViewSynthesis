from typing import List
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import Conv2d
from torch.nn import functional as F

from torchvision import models

from nn.interpolate import Downsample


class Features(Module):
    def __init__(self):
        super(Features, self).__init__()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406])[:, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225])[:, None, None],
            persistent=False,
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = (x - self.mean) / self.std
        y = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y += [x]
        return y


class VGG16(Features):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.layers = ModuleList(
            [
                features[0:4],
                features[4:9],
                features[9:16],
                features[16:23],
                features[23:30],
            ]
        )
        self.out_channels = [64, 128, 256, 512, 512]


class AlexNet(Features):
    def __init__(self):
        super(AlexNet, self).__init__()
        features = models.alexnet(pretrained=True).features
        self.layers = ModuleList(
            [
                features[0:2],
                features[2:5],
                features[5:8],
                features[8:10],
                features[10:12],
            ]
        )
        self.out_channels = [64, 192, 384, 256, 256]


class SqueezeNet(Features):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        features = models.squeezenet1_1(pretrained=True).features
        self.layers = ModuleList(
            [
                features[0:2],
                features[2:5],
                features[5:8],
                features[8:10],
                features[10:11],
                features[11:12],
                features[12:13],
            ]
        )
        self.out_channels = [64, 128, 256, 384, 384, 512, 512]


class LinLayers(ModuleList):
    def __init__(self, out_channels: List[int]):
        layers = [
            Conv2d(in_channels=o, out_channels=1, kernel_size=1, bias=False)
            for o in out_channels
        ]
        super(LinLayers, self).__init__(layers)


def get_network(net_type: str):
    if net_type == "vgg":
        return VGG16()
    elif net_type == "alex":
        return AlexNet()
    elif net_type == "squeeze":
        return SqueezeNet()
    else:
        raise NotImplementedError("choose net_type from [alex, vgg].")


def get_state_dict(net: str = "alex", version: str = "0.1"):
    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        + f"master/lpips/weights/v{version}/{net}.pth"
    )

    old_state_dict = torch.hub.load_state_dict_from_url(
        url=url,
        progress=True,
        map_location=lambda storage, loc: storage,
    )

    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        key = key.replace("lin", "")
        key = key.replace("model.1.", "")
        new_state_dict[key] = val

    return new_state_dict


class LPIPS(Module):
    def __init__(self, net: str = "alex"):
        super(LPIPS, self).__init__()

        self.net = get_network(net)
        self.lin = LinLayers(self.net.out_channels)
        self.lin.load_state_dict(get_state_dict(net, version="0.1"))
        self.freeze()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: Tensor, target: Tensor):
        inputs, targets = self.net(input), self.net(target)

        distances = []
        for l, i, t in zip(self.lin, inputs, targets):
            i = F.normalize(i, p=2, dim=-3)
            t = F.normalize(t, p=2, dim=-3)
            # distance = l(i - t).square().mean(dim=[-3, -2, -1])  # This is the formula implemented in the paper.
            distance = l((i - t).square()).mean(dim=[-3, -2, -1])
            distances += [distance]
        distances = torch.stack(distances, dim=-1)
        distances = distances.sum(dim=-1)

        return distances


class LPIPSLoss(Module):
    def __init__(self, m: int = 3):
        super(LPIPSLoss, self).__init__()
        assert m >= 1

        self.m = m
        self.criterion = LPIPS(net="vgg")
        self.interpolate = Downsample(scale_factor=0.5)
        self.freeze()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert not target.requires_grad
        losses = []
        for j in range(self.m):
            if j > 0:
                input, target = self.interpolate(input), self.interpolate(target)
            loss = self.criterion(input, target).mean()
            losses += [loss]
        losses = torch.stack(losses)
        return losses.sum()


class VGGLoss(Module):
    def __init__(self, m: int = 3):
        super(VGGLoss, self).__init__()
        assert m >= 1

        self.m = m
        self.model = VGG16()
        self.interpolate = Downsample(scale_factor=0.5)
        self.weights = [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** 0]
        self.freeze()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert not target.requires_grad
        losses = []
        for j in range(self.m):
            if j > 0:
                input, target = self.interpolate(input), self.interpolate(target)
            inputs, targets = self.model(input), self.model(target)
            for (w, i, t) in zip(self.weights, inputs, targets):
                i = F.normalize(i, p=2, dim=-3)
                t = F.normalize(t, p=2, dim=-3)
                loss = w * F.mse_loss(input=i, target=t)
                losses += [loss]
        losses = torch.stack(losses)
        return losses.sum()
