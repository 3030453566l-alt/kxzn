import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        mc = torch.sigmoid(avg_out + max_out)
        x = x * mc
        # Spatial
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ms = torch.sigmoid(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        return x * ms

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6),
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18),
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1))
        ])
        self.project = nn.Sequential(nn.Conv2d(5 * out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear')
            res.append(out)
        return self.project(torch.cat(res, dim=1))

class FeatureAlignment(nn.Module):
    """ 使用可变形卷积对齐 Skip Connection 的特征 """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * 3 * 3, kernel_size=3, padding=1)
        self.dcn = DeformConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        return F.relu(self.bn(self.dcn(x, offset)))

class MultiScaleHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, num_classes, 1)
        self.branch2 = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.branch3 = nn.Conv2d(in_ch, num_classes, 5, padding=2)
    
    def forward(self, x):
        out = self.branch1(x) + self.branch2(x) + self.branch3(x)
        return out

class MultiScaleUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        # 简化版 Encoder (DoubleConv 为你在原问题中定义的模块)
        filters = [64, 128, 256, 512]
        self.enc1 = DoubleConv(in_ch, filters[0])
        self.enc2 = DoubleConv(filters[0], filters[1])
        self.enc3 = DoubleConv(filters[1], filters[2])
        self.center = ASPP(filters[2], filters[3]) # ASPP 在瓶颈处
        
        # Decoder Components
        self.align3 = FeatureAlignment(filters[2], filters[2])
        self.align2 = FeatureAlignment(filters[1], filters[1])
        self.align1 = FeatureAlignment(filters[0], filters[0])
        
        self.dec3 = DoubleConv(filters[3] + filters[2], filters[2])
        self.att3 = CBAM(filters[2])
        
        self.dec2 = DoubleConv(filters[2] + filters[1], filters[1])
        self.att2 = CBAM(filters[1])
        
        self.dec1 = DoubleConv(filters[1] + filters[0], filters[0])
        self.att1 = CBAM(filters[0])
        
        self.head = MultiScaleHead(filters[0], num_classes)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        c = self.center(self.pool(e3))
        
        # Progressive Upsampling with Alignment & CBAM
        up3 = F.interpolate(c, scale_factor=2, mode='bilinear')
        skip3 = self.align3(e3) # 特征对齐
        d3 = self.att3(self.dec3(torch.cat([up3, skip3], dim=1)))
        
        up2 = F.interpolate(d3, scale_factor=2, mode='bilinear')
        skip2 = self.align2(e2)
        d2 = self.att2(self.dec2(torch.cat([up2, skip2], dim=1)))
        
        up1 = F.interpolate(d2, scale_factor=2, mode='bilinear')
        skip1 = self.align1(e1)
        d1 = self.att1(self.dec1(torch.cat([up1, skip1], dim=1)))
        
        return torch.sigmoid(self.head(d1))
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)