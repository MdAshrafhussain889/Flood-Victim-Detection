# ============================================================
# models/architectures.py
# Architectures: AttentionUNet, MobileNetUNet
# CPU-safe. No CUDA-specific ops.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ============================================================
# Shared Building Blocks
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBNReLU(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# Attention Gate
# ============================================================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================
# Attention U-Net
# ============================================================

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = ConvBNReLU(in_channels, 64)
        self.enc2 = ConvBNReLU(64, 128)
        self.enc3 = ConvBNReLU(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBNReLU(256, 512)
        self.att3 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=128, F_l=64,  F_int=32)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64,  64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        e3_att = self.att3(g=b,  x=e3)
        d3     = self.up3(b, e3_att)
        e2_att = self.att2(g=d3, x=e2)
        d2     = self.up2(d3, e2_att)
        e1_att = self.att1(g=d2, x=e1)
        d1     = self.up1(d2, e1_att)
        return self.out_conv(d1)


# ============================================================
# MobileNetV2 Encoder + Lightweight Decoder U-Net
# ============================================================

class MobileNetUNet(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mob     = tv_models.mobilenet_v2(weights=weights)
        features = mob.features
        self.enc0 = features[0]
        self.enc1 = features[1:4]
        self.enc2 = features[4:7]
        self.enc3 = features[7:14]
        self.enc4 = features[14:19]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(160, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up4 = UpBlock(256, 96,  128)
        self.up3 = UpBlock(128, 32,  64)
        self.up2 = UpBlock(64,  24,  32)
        self.up1 = UpBlock(32,  16,  16)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def freeze_encoder(self):
        for p in self.enc0.parameters(): p.requires_grad = False
        for p in self.enc1.parameters(): p.requires_grad = False
        for p in self.enc2.parameters(): p.requires_grad = False
        for p in self.enc3.parameters(): p.requires_grad = False
        for p in self.enc4.parameters(): p.requires_grad = False
        print("[MobileNetUNet] Encoder frozen")

    def unfreeze_encoder(self):
        for p in self.parameters(): p.requires_grad = True
        print("[MobileNetUNet] Encoder unfrozen - full fine-tune")

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.up4(b,  s3)
        d3 = self.up3(d4, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)
        out = self.final_up(d1)
        return self.out_conv(out)


# ============================================================
# Factory Function
# ============================================================

def get_model(name="attention_unet"):
    if name == "attention_unet":
        model = AttentionUNet()
        print("[Model] AttentionUNet loaded")
    elif name == "mobilenet_unet":
        model = MobileNetUNet(pretrained=True)
        print("[Model] MobileNetUNet (pretrained) loaded")
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'attention_unet' or 'mobilenet_unet'.")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {n_params:,}")
    return model.to("cpu")


if __name__ == "__main__":
    for name in ["attention_unet", "mobilenet_unet"]:
        m = get_model(name)
        dummy = torch.randn(2, 3, 256, 256)
        out   = m(dummy)
        print(f"  {name} output: {out.shape}")
        print()
