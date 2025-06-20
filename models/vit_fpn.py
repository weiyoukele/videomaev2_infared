# In models/vit_fpn.py
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()  # <-- 在MMDetection中注册这个Neck
class SimpleViTFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        """
        一个简单的FPN，用于将ViT的单尺度输出转换成多尺度。
        Args:
            in_channels (int): ViT输出的特征维度 (例如 1408 for giant)
            out_channels (int): FPN输出的每个尺度的通道数 (例如 256)
            num_outs (int): 你希望输出多少个尺度的特征 (例如 4)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        # 一个1x1卷积，用于降低通道维度
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 创建多个3x3卷积层来生成不同的特征尺度
        self.convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 来自ViT的特征图，形状为 [B, C, H, W]

        Returns:
            list[torch.Tensor]: 多尺度特征图列表
        """
        # 1. 降低通道维度
        x = self.input_proj(x)

        # 2. 生成多尺度输出
        outs = []
        # 最精细的特征直接从输入获得
        outs.append(self.convs[0](x))

        # 通过下采样（池化）生成更粗糙的特征
        for i in range(1, self.num_outs):
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            outs.append(self.convs[i](x))

        return tuple(outs)
