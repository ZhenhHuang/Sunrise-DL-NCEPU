import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvs(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, kernel_size=3, activation='relu'):
        super(DoubleConvs, self).__init__()
        hidden_channel = hidden_channel or out_channel
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size, bias=False)
        self.conv2 = nn.Conv2d(hidden_channel, out_channel, kernel_size, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.activation = F.relu if activation=='relu' else F.elu
    
    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(EncoderLayer, self).__init__()
        self.conv = DoubleConvs(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(self.pool(x))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, method='transpose', clip=False):
        super(DecoderLayer, self).__init__()
        self.clip = clip
        if method == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvs(in_channel, out_channel, in_channel//2, kernel_size=kernel_size)
        
        elif method == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvs(in_channel, out_channel, kernel_size=kernel_size)
        
    def forward(self, x_enc, x_dec):
        # N C H W
        x_dec = self.upsample(x_dec)
        H = x_enc.shape[-2] - x_dec.shape[-2]
        W = x_enc.shape[-1] - x_dec.shape[-2]
        if self.clip:
            x_enc = x_enc[:, :, H//2:H//2-H, W//2:W//2-W]
        else:
            x_dec = F.pad(x_dec, pad=[W//2, W - W//2, H//2, H-H//2])
        x = torch.cat([x_enc, x_dec], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(UNet, self).__init__()
        self.conv1 = DoubleConvs(in_channel, out_channel=64)
        self.enc1 = EncoderLayer(64, 128)
        self.enc2 = EncoderLayer(128, 256)
        self.enc3 = EncoderLayer(256, 512)
        self.enc4 = EncoderLayer(512, 1024)
        
        self.dec1 = DecoderLayer(1024, 512)
        self.dec2 = DecoderLayer(512, 256)
        self.dec3 = DecoderLayer(256, 128)
        self.dec4 = DecoderLayer(128, 64)
        
        self.output = nn.Conv2d(64, out_channel, kernel_size=1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        x = self.dec1(x4, x5)
        x = self.dec2(x3, x)
        x = self.dec3(x2, x)
        x = self.dec4(x1, x)
        
        x = self.output(x)
        return x
    

if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.randn(1, 3, 32, 32).to(device)
    model = UNet().to(device)
    out = model(x)
    print(out.shape)
        
