import torch
import torchaudio

from typing import List, Callable
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp import autocast
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_, DropPath


class PreEmphasis(nn.Module):
    # from https://github.com/clovaai/voxceleb_trainer/ 
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: Tensor) -> Tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class CASP(nn.Module):
    # from https://github.com/TaoRuijie/ECAPA-TDNN/
    def __init__(self, emb_size: int, bn: bool = True) -> None:
        super(CASP, self).__init__()
        self.att = nn.Sequential(
            nn.Conv1d(emb_size * 2 * 3, emb_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),
            nn.Tanh(),
            nn.Conv1d(emb_size, emb_size * 2, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn = nn.BatchNorm1d(emb_size * 4) if bn else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, x.shape[-1]),
                              torch.std(x, dim=2, keepdim=True).repeat(1, 1, x.shape[-1])), dim=1)

        w = self.att(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn(x)
        return x


class PosConv(nn.Module):
    def __init__(self, embed_dim: int = 256, stride: int = 1) -> None:
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
            nn.ELU()
        )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        cnn_feat_token = x.transpose(1, 2)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.transpose(1, 2)
        return x


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable) -> None:
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class GlobalLocal(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.) -> None:
        super(GlobalLocal, self).__init__()
        self.net1 = nn.Sequential(
            Rearrange('b c h -> b h c'),
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        self.kernels = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net1(x)
        x = x + self.kernels(x)
        x = self.net2(x)
        x = x.transpose(1, 2)
        x = self.drop_path(x)
        return x


class LocalConvModule(nn.Sequential):
    def __init__(self, dim: int, dropout: float = 0.) -> None:
        super(LocalConvModule, self).__init__(
            Rearrange('b t d -> b d t'),
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            GLU(1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
            Rearrange('b t d -> b d t'),
        )


class GlobalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.) -> None:
        super(GlobalAttention, self).__init__()
        
        # multi-head self-attention
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # token squeeze-and-excitation attention
        self.se = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.GELU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.se(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0., peg: bool =False) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, LocalConvModule(dim, dropout)), 
                PreNorm(dim, GlobalLocal(dim, mlp_dim, dropout = dropout)),
            ]))

        self.dim_reduce = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=False),
        )
        self.position_embedding = PosConv(dim) if peg else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        res = []
        layer_num = len(self.layers)
        for idx, (attn, lm, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = lm(x) + x
            x = ff(x) + x
            if idx < 2  or layer_num - idx <= 2:
                res.append(x)
            if idx == 0:
                x = self.position_embedding(x)

        res = [self.dim_reduce(res[i]) for i in range(4)]
        x = torch.cat(res, dim=-1)
        return x


class ViT(nn.Module):
    def __init__(self, n_mels: int, kernels: List[int], strides: List[int],  dim: int, depth: int, mlp_dim: int, heads: int,
                 dim_head: int, dropout: float = 0., peg: bool = False) -> None:
        super(ViT, self).__init__()
        patch_height, patch_width = n_mels, kernels[0]
        patch_dim = patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (c h) w'),
            nn.Conv1d(patch_height, patch_dim, kernel_size=kernels[0], stride=strides[0], padding=kernels[0] // 2),
            nn.GELU(),
            nn.BatchNorm1d(patch_dim),
            nn.Conv1d(patch_dim , dim, kernel_size=kernels[1], stride=strides[1], padding=kernels[1] // 2),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            Rearrange('b d t -> b t d'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, peg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # CLS
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class SpeakerViT(nn.Module):
    def __init__(self,
                 n_mels: int =80,
                 stem_config: dict = {"kernels":[5, 3], "strides":[1, 3]},
                 encoder_emb_size: int = 400,
                 emb_size: int = 400,
                 depth: int = 8,
                 mhsa_config: dict = {"heads": 4, "dim_head": 64},
                 feedforward_config: dict = {"mlp_dim": 800},
                 pooling_config: dict = {"emb_size": 400, "bn": True},
                 peg: bool = True,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(SpeakerViT, self).__init__()
        self.encoder = ViT(n_mels,
                          dim=encoder_emb_size,
                          depth=depth,
                          dropout=dropout,
                          peg=peg,
                          **stem_config,
                          **mhsa_config,
                          **feedforward_config,
                          **kwargs)
        self.agg = nn.Sequential(CASP(**pooling_config))
        self.fc = nn.Sequential(
            nn.Linear(encoder_emb_size * 2 * 2, emb_size),
            nn.BatchNorm1d(emb_size, affine=False)
        )

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     window_fn=torch.hamming_window, n_mels=n_mels)
                )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        with autocast(enabled=False):
            with torch.no_grad():
                x = self.torchfb(x) + 1e-6
                x = x.log()
                x = self.instancenorm(x)
                x = x.unsqueeze(1)

        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.agg(x)
        x = self.fc(x)
        return x

