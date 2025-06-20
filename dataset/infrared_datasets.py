# 在你新建的 infrared_datasets.py 文件中
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class InfraredInstanceSegmentationDataset(Dataset):
    def __init__(self, data_root, transform=None, is_train=True):
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train

        self.image_dir = os.path.join(data_root, 'images')
        self.mask_dir = os.path.join(data_root, 'masks')

        self.samples = []
        # 遍历data01-data26文件夹
        for video_folder in sorted(os.listdir(self.image_dir)):
            video_image_path = os.path.join(self.image_dir, video_folder)
            video_mask_path = os.path.join(self.mask_dir, video_folder)

            # 获取所有帧并排序
            frames = sorted(os.listdir(video_image_path))
            for frame_name in frames:
                image_path = os.path.join(video_image_path, frame_name)
                mask_path = os.path.join(video_mask_path, frame_name.replace('.jpg', '.png'))  # 假设掩码是png

                if os.path.exists(mask_path):
                    self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 这是一个简化的例子，只加载单帧。你需要修改它来加载一个视频片段(clip)
        # 真正的实现需要像 VideoClsDataset 那样采样一个16帧的片段
        image_path, mask_path = self.samples[index]

        # 1. 加载图像 (单通道)
        img = Image.open(image_path)

        # 2. 关键：通道复制以适配3通道预训练模型
        img = img.convert('RGB')

        # 3. 加载掩码图
        mask = np.array(Image.open(mask_path))  # shape: (H, W)

        # 4. 从掩码图中解析出每个实例
        # mask中的每个唯一非零值代表一个目标实例
        obj_ids = np.unique(mask)[1:]  # 忽略背景0
        num_objs = len(obj_ids)

        masks = []
        boxes = []
        labels = []

        for i in range(num_objs):
            obj_id = obj_ids[i]
            # 生成每个目标的二进制掩码
            pos = np.where(mask == obj_id)

            # 从二进制掩码计算边界框 [x_min, y_min, x_max, y_max]
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # 忽略非常小的无效框
            if xmax > xmin and ymax > ymin:
                masks.append(mask == obj_id)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # 假设只有一个类别 "target"

        # 转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        # 组装成mmdetection风格的target字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        # 应用数据增强 (必须同时作用于img和target)
        if self.transform:
            # 注意：这里的transform需要能同时处理图像和分割/检测标注
            img, target = self.transform(img, target)

        return img, target