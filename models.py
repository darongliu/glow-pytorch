import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import numpy as np


def f(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        modules.Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=True),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]), nn.ReLU(inplace=True),
        modules.Conv2dZeros(hidden_channels, out_channels))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev : (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev : (obj.shuffle(z, rev), logdet),
        "invconv1x1": lambda obj, z, logdet, rev : obj.invconv1x1(z, logdet, rev)
    }
    
    def __init__(self, in_channels, hidden_channels,
                       actnorm_scale=1.0, logscale_factor=3.0,
                       flow_permutation="invconv1x1",
                       LU_decomposed=False,
                       flow_coupling="additive"):
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.items())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale, logscale_factor)
        # 2. permute
        if flow_permutation == "invconv1x1":
            self.invconv1x1 = modules.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        C = input.size(1)
        assert C % 2 == 0
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, False)
        z1 = z[:, :C // 2, :, :]
        z2 = z[:, C // 2:, :, :]
        if self.flow_coupling == "additive":
            z2 += self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift = h[:, 0::2, :, :]
            scale = F.sigmoid(h[:, 1::2, :, :] + 2.)
            z2 += shift
            z2 *= scale
            logdet += torch.sum(torch.log(scale))
        z = torch.cat((z1, z2), dim=1)
        return z, logdet
    
    def reverse_flow(self, input, logdet):
        C = input.size(1)
        assert C % 2 == 0
        z1 = input[:, :C // 2, :, :]
        z2 = input[:, C // 2:, :, :]
        if self.flow_coupling == "additive":
            z2 -= self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift = h[:, 0::2, :, :]
            scale = F.sigmoid(h[:, 1::2, :, :] + 2.)
            z2 /= scale
            z2 -= shift
            logdet -= torch.sum(torch.log(scale))
        z = torch.cat((z1, z2), dim=1)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, True)
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, num_channels, hidden_channels, flow_steps,
                       actnorm_scale=1.0, logscale_factor=3.0,
                       flow_permutation="invconv1x1",
                       LU_decomposed=False,
                       flow_coupling="additive"):
        super().__init__()
        self.flow_steps = nn.ModuleList()
        self.flow_pools = nn.ModuleList()
        in_channels = num_channels 
        for i in range(flow_steps):
            self.flow_steps.append(FlowStep(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                actnorm_scale=actnorm_scale,
                logscale_factor=logscale_factor,
                flow_permutation=flow_permutation,
                LU_decomposed=LU_decomposed,
                flow_coupling=flow_coupling))
            if i < flow_steps - 1:
                self.flow_pools.append(modules.Split2d(
                    num_channels=in_channels))
            in_channels *= 2

    def forward(self, input, logdet=0., eps_std=None, reverse=False):
        if not reverse:
            return self.encode(input, logdet)
        else:
            return self.decode(input, eps_std)

    def encode(self, z, objective=0.0):
        for i in range(len(self.flow_steps)):
            z, objective = self.flow_steps[i](z, objective, reverse=False)
            if i < len(self.flow_pools):
                z, objective = self.flow_pools[i](z, objective, reverse=False)
        return z, objective

    def decode(self, z, eps_std=None):
        for i in reversed(range(len(self.flow_steps))):
            if i < len(self.flow_pools):
                z, _ = self.flow_pools[i](z, None, eps_std=eps_std, reverse=True)
            z, _ = self.flow_steps[i](z, None, reverse=True)
        return z


def test_flow_step():
    flow_step = FlowStep(64, 128, flow_permutation="invconv1x1").cuda()
    x = torch.Tensor(np.random.rand(4, 64, 16, 16)).cuda()
    y, _ = flow_step(x, 0)
    x_, _ = flow_step(y, 0, True)
    print(y.size(), x.size())
    print(float(torch.max(torch.abs(x - x_))))


def test_flow_net():
    flow_net = FlowNet(12, 128, 4).cuda()
    x = torch.Tensor(np.random.rand(4, 12, 16, 16)).cuda()
    y, logdet = flow_net(x, 0.0, reverse=False)
    x_, _ = flow_net(x, None, reverse=True)
    print(y.size())
    print(x_.size())


if __name__ == "__main__":
    test_flow_step()
    test_flow_net()