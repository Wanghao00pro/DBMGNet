
import os
import math
import torch
import einops
import numpy as np
import torch.nn as nn
from torch import nn, cat
from torch.nn import init
import torch.optim as optim
from einops import rearrange
# from torchsummary import summary
import torch.nn.functional as F
from models.RCGC import RCGC
from models.Embeddings import PatchEmbeddings, PositionalEmbeddings
from einops.layers.torch import Rearrange, Reduce
from models.rope_spectral import RotaryEmbedding


class Pooling(nn.Module):

    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)

class HSC(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(HSC, self).__init__()

        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=p, bias=False)

        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
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

class SEBlock(nn.Module):
    def __init__(self, planes, residual=True):
        super(SEBlock, self).__init__()
        self.cfc = nn.Conv1d(planes, planes, kernel_size=2, groups=planes) #改
        self.fc = nn.Linear(in_features=planes, out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        original_out = x
   
        mean = original_out.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = original_out.view(b, c, -1).std(-1).unsqueeze(-1)
        out = torch.cat((mean, std), -1)
        out = self.cfc(out)
        out = out.view(out.size(0), -1)
        
        weights = self.fc(out)
        activated_weights = self.sigmoid(weights)
        out = activated_weights.view(activated_weights.size(0), activated_weights.size(1), 1)

        return out

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class TIM(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(TIM, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, rgb, depth):
        
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        # out = rgb + depth
        out = self.alpha * rgb + (1 - self.alpha) * depth
        # print(self.alpha)
        return out, rgb, depth

class DBMGNet(nn.Module):
    
    def __init__(self, channels, num_classes, image_size, patch_size: int = 1, emb_dim: int = 128, 
                 num_layers: int = 1, hidden_dim: int = 128, pool: str = "mean"):
        super().__init__()

        # Params
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = channels * patch_size ** 2
        self.act = nn.ReLU(inplace=True)

        # HSC
        self.conv1 = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.conv2 = HSC(emb_dim, emb_dim, emb_dim)
        self.conv3 = HSC(emb_dim, emb_dim, emb_dim)
        self.bn3 = nn.BatchNorm2d(emb_dim)
        self.bn4 = nn.BatchNorm2d(emb_dim)
        self.conv4 = nn.Conv2d(emb_dim, channels, kernel_size=1, bias=False)

        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)

        self.graconv = RCGC(input_dim=emb_dim, hid_dim=emb_dim, output_dim=emb_dim, n_seq=self.num_patches, p_dropout=0)
        self.patchify = Rearrange("b d c  -> b c d")

        self.dropout = nn.Dropout(0.8)
        
        from mamba_ssm import Mamba
        self.num_layers = num_layers
        self.mamba = nn.ModuleList(Mamba(emb_dim,expand=1) for i in range(num_layers))
        self.mamba_back = nn.ModuleList(Mamba(emb_dim,expand=1) for i in range(num_layers))

        self.mamba_spa = nn.ModuleList(Mamba(self.num_patches,expand=1) for i in range(num_layers))
        self.mamba_back_spa = nn.ModuleList(Mamba(self.num_patches,expand=1) for i in range(num_layers))
       
        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)
        self.classifier1 = Classifier(dim=emb_dim, num_classes=num_classes)
        self.classifier2 = Classifier(dim=emb_dim, num_classes=num_classes)
       
        self.rope = RotaryEmbedding(dim=emb_dim//2)
        self.fusion = TIM(emb_dim)
        self.mlp = Mlp(emb_dim,2*emb_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)

        self.Dconv = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim, bias=False)
        self.Wconv = nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False)
        self.seblock = SEBlock(emb_dim)   
        self.conv1d0 = nn.Conv1d(emb_dim, emb_dim, kernel_size=1)
        self.conv1d1 = nn.Conv1d(emb_dim, emb_dim, kernel_size=1) 
        self.conv1d2 = nn.Conv1d(2*emb_dim, emb_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = np.squeeze(x, axis=1)
        b, c, hw = x.shape
        x = x.reshape(b,c,int(hw**0.5),int(hw**0.5))
        _,_,h,w, = x.shape

        # HSC
        x0 = self.conv1(x)
        x1 = self.conv2(self.act(self.bn2(x0)))
        x2 = self.conv3(self.act(self.bn3(x1)))
        x = self.conv4(self.act(self.bn4(x2))) + x
        
        x = self.patch_embeddings(x)
        x3 = self.pos_embeddings(x)
        x3 = self.rope.rotate_queries_or_keys(x3)
        
        # BSEBM
        x3 = self.patchify(self.ln1(x3))
        x3 = x3.reshape(b,-1,int(hw**0.5),int(hw**0.5))
        x3 = self.Wconv(self.Dconv(x3))
        x3 = x3.reshape(b,-1,hw)
        x3 = self.patchify(x3)
        x_mm =x3
        x_mm_spa = self.patchify(x3)
        for i in range(self.num_layers):
            x_m = self.mamba[i](x_mm) 
            x_m_back = self.mamba_back[i](torch.flip(x_mm,[1])) 
        for i in range(self.num_layers):
            x_m_spa = self.mamba_spa[i](x_mm_spa) 
            x_m_back_spa = self.mamba_back_spa[i](torch.flip(x_mm_spa,[1])) 
            x_mm_spa = x_m_spa + torch.flip(x_m_back_spa,[1]) 

        orial = self.patchify(x_mm_spa)
        orial = orial.reshape(b,-1,int(hw**0.5),int(hw**0.5))
        score = self.seblock(orial)

        x_m = self.patchify(x_m) * score
        x_m_back = self.patchify(torch.flip(x_m_back,[1])) * score
        x_m = self.conv1d0(x_m)
        x_m_back = self.conv1d1(x_m_back)
        x_mm = torch.cat((x_m, x_m_back), dim=1)
        x_mm = self.patchify(self.conv1d2(x_mm))
        
        x_t = x_mm 
        
        # RCGC
        x0 = self.ln(x)
        x_g = self.graconv(x0) + x
        x_c = self.mlp(self.ln(x_g)) + x_g

        # TIM
        x_t = self.patchify(x_t)
        x_t = x_t.reshape(b,self.emb_dim,int(hw**0.5),int(hw**0.5))
        x_c = self.patchify(x_c)
        x_c = x_c.reshape(b,self.emb_dim,int(hw**0.5),int(hw**0.5))
        x, x_t, x_c = self.fusion(x_t, x_c)
        x = x.reshape(b,self.emb_dim,hw)
        x = self.patchify(x)

        x_t = x_t.reshape(b,self.emb_dim,hw)
        x_t = self.patchify(x_t)
        x_c = x_c.reshape(b,self.emb_dim,hw)
        x_c = self.patchify(x_c)
        out1 = self.pool(self.dropout(x_t))
        out1 = self.classifier1(out1)
        out2 = self.pool(self.dropout(x_c))
        out2 = self.classifier2(out2)

        x = self.pool(self.dropout(x))
        x = self.classifier(x)
        
        return  x, out1, out2
