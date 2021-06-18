import torch
from torch.optim import lr_scheduler
import math
import matplotlib.pyplot as plt
from optim.sam import SAM
from torchvision.models import AlexNet

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

LR = 5e-3
DECAY = 0
EPOCH = 100

model = AlexNet(num_classes=2)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=0.5, adaptive=True, lr=LR, momentum=0.9, weight_decay=DECAY)
lf = one_cycle(1, 0.3, EPOCH)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scheduler.last_epoch = -1

plt.figure()
x = list(range(100))
y = []

for epoch in range(EPOCH):
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    print(epoch,'lr',lr)
    y.append(scheduler.get_last_lr()[0])

plt.xlabel("epoch")
plt.ylabel("lr")
plt.plot(x,y)
plt.show()