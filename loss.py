import torch.nn as nn
import torch
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np

# BraTS = 4 label
cfs = ConfusionMatrix(num_classes=4)

class DiceLoss_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)
    
    def binary_dice(self, inputs, targets, smooth=1):
        # print(f"targets shape: {targets.shape}")
        # print(f"inputs shape: {inputs.shape}")
        
        # targets = targets.unsqueeze(-1).expand(-1, -1, -1, 4) # 클래스 차원 추가
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1-dice

    def forward(self, ipts, gt):
        
        ipts = self.sfx(ipts)
        c = ipts.shape[1]
        sum_loss = 0
        for i in range(c):
            tmp_inputs = ipts[:,i]
            tmp_gt = gt[:,i]
            tmp_loss = self.binary_dice(tmp_inputs,tmp_gt)
            sum_loss += tmp_loss
        return sum_loss / c
    
class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        # inputs = self.sfx(inputs)
        inputs = self.sfx(inputs.float())
        c = inputs.shape[1]
        # print(f"targets shape: {targets.shape}")
        # print(f"inputs shape: {inputs.shape}")
        # print(c)
        inputs = torch.max(inputs,1).indices.cpu()
        targets = torch.max(targets,1).indices.cpu()
        # inputs = torch.clamp(inputs, min=0, max=c-1)
        # targets = torch.clamp(targets, min=0, max=c-1)
        # targets = targets.contiguous().unsqueeze(-1).expand(-1, -1, -1, 4) # 클래스 차원 추가
        cfsmat = cfs(inputs,targets).numpy()
        
        sum_iou = 0
        for i in range(c):
            tp = cfsmat[i,i]
            '''
            [0:3,i]로 하면 0, 1, 2의 클래스만 확인
            '''
            fp = np.sum(cfsmat[:, i]) - tp  # False Positives
            fn = np.sum(cfsmat[i, :]) - tp  # False Negatives
            # fp = np.sum(cfsmat[0:3,i]) - tp
            # fn = np.sum(cfsmat[i,0:3]) - tp
        
            tmp_iou = tp / (fp + fn + tp)
            sum_iou += tmp_iou
                
        return sum_iou / c

class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)
        # print(f"inputs shape: {inputs.shape}")
        # targets = targets.unsqueeze(-1).expand(-1, -1, -1, -1, 4) # 클래스 차원 추가
        # print(f"targets shape: {targets.shape}")
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1-dice

class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
        # targets = targets.unsqueeze(-1).expand(-1, -1, -1, -1, 4) # 클래스 차원 추가
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU
