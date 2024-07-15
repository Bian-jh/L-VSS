import torch
import torch.nn as nn

import math
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerOp(nn.Module):
    def __init__(
            self, in_chans, out_chans, stride: int = 1, padding: int = 0, dilation: int = 1,
            bias: bool = True, dimension: str = ''
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        assert self.in_chans == self.out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        Token Mixer along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f"{self.dimension} dimension not implemented")
        return deform_conv2d(
            input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation
        )


class Token_Mixer(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim

        self.tm_c = nn.Linear(dim, dim, bias=False)
        self.tm_h = MixerOp(dim, dim, dimension='h')
        self.tm_w = MixerOp(dim, dim, dimension='w')

        self.fusion = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        output: [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f"offset shape not match, got {offset.shape}"
        w = self.tm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :]).permute(0, 2, 3, 1)
        h = self.tm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :]).permute(0, 2, 3, 1)
        c = self.tm_c(x)

        a = (w + h + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = w * a[0] + h * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


