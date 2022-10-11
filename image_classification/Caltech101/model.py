import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


def image2patch(x, patch_size=4):
    # B, H, W, C -> B, L, D
    B, H, W, C = x.shape
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, patch_size * patch_size * C)
    return x


def patch2image(x, H, W, patch_size=4):
    # B, L, D -> B, H, W, C
    B, L, D = x.shape
    x = x.reshape(B, H//patch_size, W//patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x


class PatchMerging(nn.Module):
    def __init__(self, C):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * C)
        self.proj = nn.Linear(4 * C, 2 * C, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.proj(self.norm(x))
        return x.reshape(B, H//2, W//2, -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation='GELU', drop=0.1):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = F.gelu if activation == 'GELU' else F.relu
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.activation(self.linear1(x)))
        x = self.drop(self.linear2(x))
        return x


class FourierPatchBlock(nn.Module):
    def __init__(self, H, W, in_channels, out_channels):
        super(FourierPatchBlock, self).__init__()
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, dtype=torch.cfloat))
        self.H = H
        self.W = W

    def forward(self, x):
        # B, L, C
        B, _, C = x.shape
        x = x.reshape(B, x.shape[1], 8, -1)    # B, L, H, D
        x = torch.fft.rfft(x, dim=1)
        x = torch.einsum('blhd,hdo->blho', x, self.weights)
        x = torch.fft.irfft(x, dim=1).reshape(B, -1, C)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        # N C H W
        x = self.conv1(x) + self.conv2(x)
        x = F.relu(x)
        return x


class IdentityBlock(nn.Module):
    def __init__(self, channels):
        super(IdentityBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.Conv2d(channels // 2, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        x = F.relu(x + self.conv(x))
        return x


class PatchEnhanceLayer(nn.Module):
    def __init__(self, H, W, in_features, hidden_features=None, out_features=None, activation='GELU', drop=0.1):
        super(PatchEnhanceLayer, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.block1 = FourierPatchBlock(H, W, in_features, in_features)
        self.mlp = Mlp(in_features, hidden_features, out_features, activation, drop)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

    def forward(self, x):
        # B, L, D
        x = self.block1(x) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)
        return x


class Model(nn.Module):
    def __init__(self, H, W, c_in, C, n_classes=102, patch_size=4, res_layers=1, drop=0.1):
        super(Model, self).__init__()
        # C = 48
        self.block1 = nn.Sequential(
            nn.Linear(c_in * patch_size * patch_size, C * patch_size * patch_size),
            PatchEnhanceLayer(H, W, C * patch_size * patch_size, drop=drop)
        )
        self.merge = PatchMerging(C)

        self.block2 = nn.Sequential(
            IdentityBlock(2 * C),
            ConvBlock(2 * C, 4 * C),
            IdentityBlock(4 * C),
            nn.AvgPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            IdentityBlock(4 * C),
            ConvBlock(4 * C, 8 * C),
            IdentityBlock(8 * C),
            nn.AvgPool2d(2, 2),
        )

        self.block4 = nn.Sequential(
            IdentityBlock(8 * C),
            ConvBlock(8 * C, 16 * C),
            IdentityBlock(16 * C),
            nn.AvgPool2d(2, 2),
        )

        self.block5 = nn.Sequential(
            IdentityBlock(16 * C),
            ConvBlock(16 * C, 32 * C),
            IdentityBlock(32 * C),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(32 * C, n_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, H, W, C = x.shape
        x = image2patch(x)
        x = self.block1(x)
        x = patch2image(x, H, W)
        x = self.merge(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.pool(x)
        x = nn.Flatten()(x)
        x = self.proj(x)
        return x


class DeepWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(DeepWiseConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=(kernel-1)//2, groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels//2, 1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, 1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels//2)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, factor=16):
        super(SqueezeExcitation, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(in_channels, in_channels//factor)
        self.linear2 = nn.Linear(in_channels//factor, in_channels)

    def forward(self, x):
        #res = x
        x = self.avg(x).squeeze()
        x = F.relu(self.linear1(x))
        x = F.hardsigmoid(self.linear2(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        #x = res * x.unsqueeze(2).unsqueeze(2)
        return x


class DeepWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, factor=16):
        super(DeepWiseSeparableConv, self).__init__()
        self.dw = DeepWiseConv(in_channels, out_channels, kernel)
        self.se = SqueezeExcitation(out_channels, factor)

    def forward(self, x):
        x = self.dw(x)
        x = self.se(x)
        return x


class DPSENet(nn.Module):
    def __init__(self, in_channels=3, c_out=102):
        super(DPSENet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dsconv1 = nn.Sequential(
            DeepWiseSeparableConv(64, 128, 3),
            DeepWiseSeparableConv(128, 128, 3),
            nn.MaxPool2d(2, 2),
        )
        self.dsconv2 = nn.Sequential(
            DeepWiseSeparableConv(128, 256, 3),
            DeepWiseSeparableConv(256, 256, 3),
            nn.MaxPool2d(2, 2),
        )
        self.dsconv3 = nn.Sequential(
            DeepWiseSeparableConv(256, 512, 3),
            DeepWiseSeparableConv(512, 512, 3),
            nn.MaxPool2d(2, 2),
        )
        self.dsconv4 = nn.Sequential(
            DeepWiseSeparableConv(512, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            DeepWiseSeparableConv(1024, 1024, 5),
            nn.MaxPool2d(2, 2),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, c_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.avg(x)
        x = nn.Flatten()(x)
        x = F.hardsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class DistilLoss(nn.Module):
    def __init__(self, T, alpha):
        super(DistilLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.KLDivLoss()

    def forward(self, y1, y2, y):
        loss1 = self.loss1(y1, y)
        st = torch.softmax(y1 / self.T, dim=-1)
        tch = torch.softmax(y2 / self.T, dim=-1)
        loss2 = self.loss2(st, tch)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss


def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),
                
                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish(),
                
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish()
            )

    def forward(self, x):
        return self.conv(x)


class PPLCNet(nn.Module):
    def __init__(self, scale=1.0, num_classes=1000, dropout_prob=0.2):
        super(PPLCNet, self).__init__()
        self.cfgs = [
           # k,  c,  s, SE
            [3,  32, 1, 0],

            [3,  64, 2, 0],
            [3,  64, 1, 0],
            
            [3,  128, 2, 0],
            [3,  128, 1, 0],
            
            [5,  256, 2, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            
            [5,  512, 2, 1],
            [5,  512, 1, 1],
        ]
        self.scale = scale

        input_channel = _make_divisible(16 * scale)
        layers = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), HardSwish()]

        block = DepSepConv
        for k, c, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(input_channel, 1280, 1, 1, 0)
        self.hwish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
                

class SpinalNet_ResNet(nn.Module):
    def __init__(self, feature_in, layer_width):
        super(SpinalNet_ResNet, self).__init__()
        self.feature_in = feature_in
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Linear(feature_in, layer_width),
            nn.ReLU()
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Linear(feature_in+layer_width, layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(feature_in+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(feature_in+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, Num_class),)
        
    def forward(self, x):
        feature_in = self.feature_in
        x1 = self.fc_spinal_layer1(x[:, 0:feature_in])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,feature_in:2*feature_in], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:feature_in], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,feature_in:2*feature_in], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)
        return x
        

if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.randn(1, 128, 128, 3).to(device)
    model = Model(128, 128, 3, 48).to(device)
    x = model(x)
    print(x.shape)





















