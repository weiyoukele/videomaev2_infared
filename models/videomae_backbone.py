# In models/videomae_backbone.py
import torch
from mmdet.models.builder import BACKBONES
from .modeling_finetune import VisionTransformer  # 从你的代码库中导入


@BACKBONES.register_module()  # <-- 在MMDetection中注册这个Backbone
class VideoMAEBackbone(VisionTransformer):
    def __init__(self, **kwargs):
        # 从kwargs中移除mmdet不认识的参数
        kwargs.pop('pretrained', None)
        super().__init__(**kwargs)
        # 移除原来的分类头
        self.head = torch.nn.Identity()

    def forward(self, x):
        # mmdet的输入是 (B, C, H, W)，而我们的模型是 (B, C, T, H, W)
        # 我们需要处理这个时间维度。一个简单的方法是，将输入在时间上堆叠
        # 假设输入x是 [B, C, T, H, W]

        B, C, T, H, W = x.shape

        # 调用原始的forward_features来获取序列特征
        # [B, NumPatches, EmbedDim]
        features_seq = self.forward_features(x)

        # 将序列特征Reshape回2D特征图的形状
        T_p = T // self.patch_embed.tubelet_size
        H_p = H // self.patch_embed.patch_size[0]
        W_p = W // self.patch_embed.patch_size[1]

        # [B, T_p, H_p, W_p, C]
        features_map_5d = features_seq.reshape(B, T_p, H_p, W_p, self.embed_dim)

        # 为了送入FPN，我们需要一个4D张量。这里简单地对时间维度做平均池化
        # 这是最简单的方法，更复杂的方法可以尝试3D卷积或保留时间维度
        features_map_4d = torch.mean(features_map_5d, dim=1)  # -> [B, H_p, W_p, C]

        # 转换为 [B, C, H, W] 格式
        features_map_4d = features_map_4d.permute(0, 3, 1, 2)

        # FPN期望一个特征列表，我们这里只返回一个单尺度特征
        return [features_map_4d]

    def init_weights(self, pretrained=None):
        if pretrained:
            print(f"Loading pretrained weights from: {pretrained}")
            checkpoint = torch.load(pretrained, map_location='cpu')
            # 处理权重key不匹配的问题（例如 'model.' 前缀）
            checkpoint_model = checkpoint.get('model', checkpoint)
            msg = self.load_state_dict(checkpoint_model, strict=False)
            print("Pretrained weights loading message:", msg)
        else:
            # 如果没有预训练权重，则执行原始的权重初始化
            super()._init_weights()
