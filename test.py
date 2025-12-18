from attention_unet import AttU_Net
from MultiScaleUNet import MultiScaleUNet
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# 已训练模型权重路径
MODEL_PATH = r"./saved_model/multiUnet_1_liver_21.pth"

# 加载模型并设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleUNet(3, 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 加载和预处理图像
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),         # 转为Tensor
        transforms.Normalize([0.5], [0.5])  # 归一化: 均值0.5, 标准差0.5
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # 增加一个批量维度

# 读取图像
input_image_path = r"./dataset/liver/val/000.png"  # 替换为你的输入图像路径
input_image = load_image(input_image_path).to(device)

# 推理
output = model(input_image)

# 后处理：将输出转换为图像格式
output = output.squeeze(0).squeeze(0).cpu().detach().numpy()  # 转为单通道2D数组
output = (output > 0.5).astype(np.uint8)  # 二值化：阈值0.5

# 显示输入图像和预测结果
def display_images(input_tensor, output_array):
    # 将输入图像从 Tensor 转换为 numpy 格式
    input_image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC 格式
    input_image = (input_image * 0.5 + 0.5)  # 反归一化

    # 输出图像
    plt.figure(figsize=(10, 5))
    # 原始输入
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_image)
    # 输出推理结果
    plt.subplot(1, 2, 2)
    plt.title("Output Segmentation")
    plt.imshow(output_array, cmap="gray")
    plt.savefig("output_segmentation.png")
    plt.show()

display_images(input_image, output)