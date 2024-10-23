import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from net import GCPANet  # 假设GCPANet已经实现
import model as modellib  # 假设Mask-RCNN已经实现
from config import Config
import numpy as np

# GCPANet 和 Mask-RCNN 联合模型定义
class GCPANet_MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(GCPANet_MaskRCNN, self).__init__()
        
        # Initialize GCPANet
        self.gcpa_net = GCPANet()

        # Initialize Mask-RCNN with a ResNet50 backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉ResNet的最后两层
        backbone.out_channels = 2048

        # Initialize Mask-RCNN
        self.mask_rcnn = modellib.MaskRCNN(backbone, num_classes=num_classes)

    def forward(self, images, masks, targets=None):
        # Step 1: 使用 GCPANet 分割肺实质
        lung_mask = self.gcpa_net(images)  # 得到肺实质分割掩码

        # Step 2: 使用分割掩码将非肺实质区域屏蔽
        combined_input = images * lung_mask

        # Step 3: 使用 Mask-RCNN 进行病灶检测和分割
        if self.training:
            losses = self.mask_rcnn(combined_input, targets)
            return losses
        else:
            predictions = self.mask_rcnn(combined_input)
            return predictions


# 配置类
class CustomConfig(Config):
    NAME = "lung_lesion"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # 一个病灶类别加背景
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    DETECTION_MIN_CONFIDENCE = 0.7  # 检测置信度阈值


# 自定义数据集类，用于加载肺实质分割掩码和病灶标签
class LungLesionDataset(COCO):
    def load_data(self, dataset_dir, subset):
        """加载COCO格式数据集，准备肺实质掩码和病灶标签"""
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, "2014"))
        self.load_coco(coco, dataset_dir, subset)
        self.prepare()

    def load_mask(self, image_id):
        """加载实例掩码，返回肺实质掩码和病灶标签"""
        image_info = self.image_info[image_id]

        # 使用父类的load_mask方法加载病灶标签
        lesion_mask, class_ids = super().load_mask(image_id)

        # 加载预先生成的肺实质分割掩码
        lung_mask = ...  # 你需要从相应文件夹加载预先生成的肺实质掩码

        # 合并病灶掩码和肺实质掩码
        combined_mask = lesion_mask * lung_mask  # 只保留肺实质区域的病灶掩码
        return combined_mask, class_ids


# 训练函数
def train_model(dataset_train, dataset_val, config):
    # 初始化 GCPANet 和 Mask-RCNN 联合模型
    model = GCPANet_MaskRCNN(num_classes=config.NUM_CLASSES)
    model = model.cuda()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 开始训练
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0

        # 遍历数据集
        for images, masks, targets in DataLoader(dataset_train, batch_size=config.IMAGES_PER_GPU, shuffle=True):
            images = images.cuda()
            masks = masks.cuda()
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # 前向传播计算损失
            losses = model(images, masks, targets)
            loss = sum(loss for loss in losses.values())

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {epoch_loss}")

        # 每个epoch结束后进行验证
        evaluate_coco(model, dataset_val)


# 验证函数
def evaluate_coco(model, dataset_val):
    """在验证集上评估模型性能"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks, targets in DataLoader(dataset_val, batch_size=1, shuffle=False):
            images = images.cuda()
            masks = masks.cuda()
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # 前向传播进行推理
            losses = model(images, masks, targets)
            loss = sum(loss for loss in losses.values())
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss}")


# 主函数
if __name__ == '__main__':
    # 初始化配置
    config = CustomConfig()
    config.EPOCHS = 50  # 设置训练的epoch数量

    # 加载训练集
    dataset_train = LungLesionDataset()
    dataset_train.load_data('/path/to/dataset', 'train')

    # 加载验证集
    dataset_val = LungLesionDataset()
    dataset_val.load_data('/path/to/dataset', 'val')

    # 开始训练模型
    train_model(dataset_train, dataset_val, config)
