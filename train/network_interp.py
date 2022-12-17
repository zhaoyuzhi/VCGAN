import torch
from collections import OrderedDict

alpha = 0.3  # interpolation coefficient
net_A = torch.load('model1_Second_Stage_epoch500_bs1_448p.pth')
net_B = torch.load('model2_Second_Stage_epoch500_bs1_448p.pth')

net_interp = OrderedDict()
for k, v_A in net_A.items():
    v_B = net_B[k]
    net_interp[k] = alpha * v_A + (1 - alpha) * v_B

print(net_interp)
