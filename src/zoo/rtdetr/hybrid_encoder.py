'''
by lyuwenyu
Modified to include CBAM (Convolutional Block Attention Module)
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from src.core import register

__all__ = ['HybridEncoder']

# ==================== CBAM 模組定義 ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
# =======================================================

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

@register
class HybridEncoder(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], hidden_dim=256, use_encoder_idx=[2], num_encoder_layers=1, nhead=8, dim_feedforward=1024, dropout=0., activation='gelu', expansion=1.0, depth_mult=1.0, act='silu'):
        super().__init__()
        self.in_channels = in_channels
        self.use_encoder_idx = use_encoder_idx
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(ConvNormLayer(ch, hidden_dim, 1, 1, act=act))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_encoder_layers) for _ in range(len(use_encoder_idx))])

        # Top-down FPN blocks
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        # [新增] FPN CBAM 模組
        self.fpn_cbams = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(RepVggBlock(hidden_dim * 2, hidden_dim, act=act))
            self.fpn_cbams.append(CBAM(hidden_dim)) # 為每個 FPN 融合層添加 CBAM

        # Bottom-up PAN blocks
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        # [新增] PAN CBAM 模組
        self.pan_cbams = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(RepVggBlock(hidden_dim * 2, hidden_dim, act=act))
            self.pan_cbams.append(CBAM(hidden_dim)) # 為每個 PAN 融合層添加 CBAM

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        # 1. Input projection and Transformer Encoder
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.encoder is not None:
            for i, idx in enumerate(self.use_encoder_idx):
                feat = proj_feats[idx]
                b, c, h, w = feat.shape
                feat = feat.flatten(2).permute(0, 2, 1)
                feat = self.encoder[i](feat)
                proj_feats[idx] = feat.permute(0, 2, 1).reshape(b, c, h, w)

        # 2. Broadcasting and fusion (Top-down FPN)
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # 通過 FPN Block
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            
            # ===== [新增] 應用 CBAM 注意力機制 =====
            inner_out = self.fpn_cbams[len(self.in_channels)-1-idx](inner_out)
            
            inner_outs.insert(0, inner_out)

        # 3. Bottom-up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            
            # 通過 PAN Block
            cur_feat = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            
            # ===== [新增] 應用 CBAM 注意力機制 =====
            cur_feat = self.pan_cbams[idx](cur_feat)
            
            outs.append(cur_feat)

        return outs