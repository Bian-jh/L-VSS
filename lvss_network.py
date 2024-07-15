import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from lib.conv import Conv
from lib.fusion import aggregation
from lib.patch_mixer import Token_Mixer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Conv_stem(nn.Module):
    def __init__(self, in_chans=3, hidden_dim=32, out_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # b, out_dim, h/4, w/4

        tokens = x.permute(0, 2, 3, 1)  # b, h/4, w/4, out_dim
        return tokens


class Stage(nn.Module):
    def __init__(self, num_blocks, out_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            blocks.append(Block(out_dim=out_dim, drop_path=drop_path[j], norm_layer=norm_layer))

        self.blocks = nn.ModuleList(blocks)

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

    def forward(self, tokens):
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens


class Block(nn.Module):
    def __init__(self, out_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.local_ss2d = Local_SS2D(out_dim)
        self.ss2d = SS2D(d_model=out_dim, dropout=0, d_state=16)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, tokens):
        tokens = tokens + self.drop_path(self.ss2d(self.local_ss2d(tokens)))

        return tokens


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):  # (b,h,w,c)->(b,h/2,w/2,2c)
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):  #(b,h,w,c)->(b,h,w,2c)->(b,2h,2w,c/2)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x).permute(0, 3, 1, 2)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class Local_SS2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1, bias=False, groups=dim // 4)
        self.dw2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, bias=False, groups=dim // 4)
        self.post_conv = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.se_cnn = SELayer(channel=dim // 2, reduction=4)
        self.se_mamba = SELayer(channel=dim // 2, reduction=4)
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.SS2D = SS2D(d_model=dim // 2, dropout=0, d_state=16)
        self.increase_dim = nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_norm(x)
        x = x.permute(0, 3, 1, 2)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x11, x12 = torch.chunk(x1, 2, dim=1)
        x11 = self.dw1(x11)
        x12 = self.dw2(x12)

        x1 = x11 + x12
        x1 = self.post_conv(x1)

        x2 = x2.permute(0, 2, 3, 1)

        x2 = self.SS2D(x2)

        x2 = x2.permute(0, 3, 1, 2)

        x = x2 * torch.sigmoid(self.se_cnn(x1) + self.se_mamba(x2)).expand_as(x2)
        x = self.increase_dim(x)

        x = x.permute(0, 2, 3, 1)
        x = self.post_norm(x)
        return x


class Mamba_extractor(nn.Module):
    def __init__(self, drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        depths = [2, 4, 9, 4]
        out_dims = [64, 64 * 2, 64 * 4, 64 * 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embedding = Conv_stem()

        depth = 0
        self.patch_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.patch_merges.append(PatchMerging2D(out_dims[i - 1]))
            self.stages.append(Stage(depths[i], out_dim=out_dims[i], drop_path=dpr[depth:depth + depths[i]], norm_layer=norm_layer))
            depth += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        tokens = self.patch_embedding(x)

        outputs = []

        for i in range(4):
            if i > 0:
                tokens = self.patch_merges[i - 1](tokens)
            tokens = self.stages[i](tokens)
            mid_out = tokens.permute(0, 3, 1, 2)

            if i > 0:
                outputs.append(mid_out)

        return outputs


class SDNet(nn.Module):
    def __init__(self, channel=32, share_dim=1):
        super().__init__()
        self.share_dim = share_dim
        self.feature_extractor = Mamba_extractor()
        self.agg = aggregation(channel=channel)

        self.trans1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.trans2 = Conv(256, 32, 3, 1, padding=1, bn_acti=True)
        self.trans3 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)

        self.offset_layer1 = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32 * 2 // self.share_dim)
        )
        self.offset_layer2 = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32 * 2 // self.share_dim)
        )
        self.offset_layer3 = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32 * 2 // self.share_dim)
        )

        self.token_mixer1 = Token_Mixer(32)
        self.token_mixer2 = Token_Mixer(32)
        self.token_mixer3 = Token_Mixer(32)

        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

    def forward(self, x):
        _, _, h, w = x.shape

        outputs1, outputs2, outputs3 = self.feature_extractor(x)

        agg_feature = self.agg(outputs3, outputs2, outputs1)
        pre_map_4 = F.interpolate(agg_feature, scale_factor=8, mode='bilinear')

        outputs1 = self.trans1(outputs1).permute(0, 2, 3, 1)
        outputs2 = self.trans2(outputs2).permute(0, 2, 3, 1)
        outputs3 = self.trans3(outputs3).permute(0, 2, 3, 1)

        offset1 = self.offset_layer1(outputs1).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2)
        offset2 = self.offset_layer2(outputs2).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2)
        offset3 = self.offset_layer3(outputs3).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2)

        # ------------------- attention-one -----------------------
        up_fea3 = F.interpolate(agg_feature, scale_factor=0.25, mode='bilinear')
        up_fea3_ra = -1 * (torch.sigmoid(up_fea3)) + 1
        mixer_3 = self.token_mixer3(outputs3, offset3).permute(0, 3, 1, 2)
        atten_3 = up_fea3_ra.expand(-1, 32, -1, -1).mul(mixer_3)
        ra_3 = self.ra3_conv1(atten_3)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3)

        x_3 = ra_3 + up_fea3
        pre_map_3 = F.interpolate(x_3, scale_factor=32, mode='bilinear')

        # ------------------- attention-two -----------------------
        up_fea2 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        up_fea2_ra = -1 * (torch.sigmoid(up_fea2)) + 1
        mixer_2 = self.token_mixer2(outputs2, offset2).permute(0, 3, 1, 2)
        atten_2 = up_fea2_ra.expand(-1, 32, -1, -1).mul(mixer_2)
        ra_2 = self.ra2_conv1(atten_2)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)

        x_2 = ra_2 + up_fea2
        pre_map_2 = F.interpolate(x_2, scale_factor=16, mode='bilinear')

        # ------------------- attention-three -----------------------
        up_fea1 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        up_fea1_ra = -1 * (torch.sigmoid(up_fea1)) + 1
        mixer_1 = self.token_mixer1(outputs1, offset1).permute(0, 3, 1, 2)
        atten_1 = up_fea1_ra.expand(-1, 32, -1, -1).mul(mixer_1)
        ra_1 = self.ra1_conv1(atten_1)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)

        x_1 = ra_1 + up_fea1
        pre_map_1 = F.interpolate(x_1, scale_factor=8, mode='bilinear')

        pre_map = pre_map_1 + pre_map_2 + pre_map_3 + pre_map_4

        return pre_map, pre_map_1, pre_map_2, pre_map_3, pre_map_4


