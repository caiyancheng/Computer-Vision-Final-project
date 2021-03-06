from os import X_OK
import torch 
import torchvision 
import math
from net.cvt_offical import *
from utils import train_transform, test_transform,initialize,smooth_crossentropy
from config import device
from optim.sam import SAM
from ptflops import get_model_complexity_info
import torch.nn.functional as F
from torch.optim import lr_scheduler

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

device = torch.device('cuda:2')

initialize(42)
pth = 'CvT-13-224x224-IN-1k.pth'
# model = get_cvt13_pretrained(pth).to(device)
#model = BiseCvt_same_flops(pth).to(device)#4.12FLOPS 23.53param
#model = Cvt_same_params3(pth,hidden_dim=256,up_factor=4).to(device)
#model = BiseCvt_same_params(pth).to(device)
model = Cvt_same_flops2(pth,hidden_dim=16).to(device)

flops, params = get_model_complexity_info(model,(3,32,32),print_per_layer_stat=False)
print('flops',flops, 'params',params)
# flops 4.43 GMac params 23.53 M 均比resnet50小

BATCH = 48
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

LR = 5e-3
DECAY = 0
EPOCH = 100

#fine-tune by 1w step on cifar10 with  batch_size=512 ,lr = base_lrs, weight_decay=0,
#experiment setting according to Vit and Cvt paper
# base_lrs = [0.001, 0.003, 0.01, 0.03]
# epoches = 100 for 512 batch, 

#optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=DECAY)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=0.5, adaptive=True, lr=LR, momentum=0.9, weight_decay=DECAY)
lf = one_cycle(1, 0.1, EPOCH)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scheduler.last_epoch = -1


def train(dataloader,train_sam=True):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)

        #x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)        
        logits = model(x)

        #非SAM
        if train_sam is False:
            optimizer.zero_grad()
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizer.step()

        #可以考虑使用SAM进行训练
        else:
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(x), y).mean().backward()
            optimizer.second_step(zero_grad=True)

        tot_loss += loss.item()
        tot_num += x.shape[0]

    return tot_loss / tot_num

def test(dataloader):
    model.eval()
    correct = 0
    num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)
        #x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)  

        logits = model(x)
        pred = logits.argmax(1)
        correct += torch.sum(pred==y).item()
        num += x.shape[0]
    acc = correct / num
    return acc 

if __name__ == '__main__':
    accs = []
    losses = []
    model_path = 'pth/cvt_sam_flop_5e-3_cyc2_lr2_nopre.pth'#lr(1->0.3)  lr2(1->0.1)
    for epoch in range(EPOCH):
        #训练sam大叔优化器,也可以考虑使用KFAC优化器
        lr = scheduler.get_last_lr()[0]
        loss = train(trainloader)
        scheduler.step()
        testacc = test(testloader)
        accs.append(testacc)
        losses.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'lr',lr,'loss',loss,'acc',testacc)
        
        #可以考虑使用scheduler
        #scheduler(epoch)
    
        torch.save(model.state_dict(),model_path)
    