import torch
import torch.nn as nn


def activation(num_channel):
    return ASWISH(num_channel=num_channel)


class SLOPE(nn.Module):
    def __init__(self, num_channel):
        super(SLOPE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channel * 2, num_channel)
        self.bn1 = nn.BatchNorm1d(num_channel)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_channel, num_channel)
        self.bn2 = nn.BatchNorm1d(num_channel)

    def forward(self, x):
        a = torch.max(x, torch.zeros_like(x))
        a = self.avgpool(a)
        a = torch.flatten(a, 1)
        b = torch.min(x, torch.zeros_like(x))
        b = self.avgpool(b)
        b = torch.flatten(b, 1)
        c = torch.cat((a, b), dim=1)
        out = self.fc1(c)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = 1.5 * torch.sigmoid(out)
        return out


class ASWISH(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.slope1 = SLOPE(num_channel=num_channel)

    def forward(self, x):
        beta = self.slope1(x)
        beta = beta.unsqueeze(2)
        return x * torch.sigmoid(beta * x)


class SLDC(nn.Module):
    def __init__(self, num_channel, nngs_ks=63):
        super().__init__()

        self.conv1 = nn.Conv1d(
            num_channel, num_channel, kernel_size=nngs_ks, stride=1, padding=nngs_ks // 2,
            bias=False, groups=num_channel)
        self.bn = nn.BatchNorm1d(num_channel)
        self.act = activation(num_channel)
        self.conv2 = nn.Conv1d(
            num_channel, num_channel, kernel_size=nngs_ks, stride=1, padding=nngs_ks // 2,
            bias=False, groups=num_channel)

    def forward(self, x):

        lam = self.conv1(x)
        lam = self.bn(lam)
        lam = self.act(lam)
        lam = self.conv2(lam)
        lam = torch.abs(lam)

        m = torch.max(
            torch.abs(x) - lam ** 2 / (torch.abs(x) + 1e-5),
            torch.zeros_like(x))

        return m * torch.sign(x)


class NNGSBU(nn.Module):
    def __init__(
            self, in_channel, out_channel, stride,
            nngs_ks):
        super().__init__()
        if stride != 1 or in_channel != out_channel:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channel)
            self.downsample = nn.Sequential(self.conv1, self.bn1)
        else:
            self.downsample = None

        self.sldc = SLDC(num_channel=in_channel, nngs_ks=nngs_ks)

        self.bn2 = nn.BatchNorm1d(in_channel)
        self.act1 = activation(num_channel=in_channel)
        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm1d(out_channel)
        self.act2 = activation(num_channel=out_channel)
        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x

        out = self.sldc(x)

        out = self.bn2(out)
        out = self.act1(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class NNGSN_AS(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        inplanes = 4
        nngs_ks = [31] * 4

        self.conv = nn.Conv1d(in_channel, inplanes, kernel_size=31, stride=2, padding=15, bias=False)  # 512

        BasicUnit = NNGSBU

        self.layer1 = BasicUnit(
            inplanes, inplanes, stride=2, nngs_ks=nngs_ks[0])

        self.layer2_1 = BasicUnit(
            inplanes, inplanes, stride=1, nngs_ks=nngs_ks[1])
        self.layer2_2 = BasicUnit(
            inplanes, inplanes, stride=1, nngs_ks=nngs_ks[1])
        self.layer2_3 = BasicUnit(
            inplanes, inplanes, stride=1, nngs_ks=nngs_ks[1])
        self.layer3 = BasicUnit(
            inplanes, inplanes * 2, stride=2, nngs_ks=nngs_ks[1])

        self.layer4_1 = BasicUnit(
            inplanes * 2, inplanes * 2, stride=1, nngs_ks=nngs_ks[2])
        self.layer4_2 = BasicUnit(
            inplanes * 2, inplanes * 2, stride=1, nngs_ks=nngs_ks[2])
        self.layer4_3 = BasicUnit(
            inplanes * 2, inplanes * 2, stride=1, nngs_ks=nngs_ks[2])
        self.layer5 = BasicUnit(
            inplanes * 2, inplanes * 4, stride=2, nngs_ks=nngs_ks[2])

        self.layer6_1 = BasicUnit(
            inplanes * 4, inplanes * 4, stride=1, nngs_ks=nngs_ks[3])
        self.layer6_2 = BasicUnit(
            inplanes * 4, inplanes * 4, stride=1, nngs_ks=nngs_ks[3])
        self.layer6_3 = BasicUnit(
            inplanes * 4, inplanes * 4, stride=1, nngs_ks=nngs_ks[3])

        self.bn = nn.BatchNorm1d(inplanes * 4)
        self.act = activation(num_channel=inplanes * 4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(inplanes * 4, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        x = self.layer3(x)

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)

        x = self.layer5(x)

        x = self.layer6_1(x)
        x = self.layer6_2(x)
        x = self.layer6_3(x)

        x = self.bn(x)
        x = self.act(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

