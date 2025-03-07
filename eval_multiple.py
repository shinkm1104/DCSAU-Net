import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import multi_classes
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
from pytorch_lightning.metrics import ConfusionMatrix
import os 
import cv2
import math

os.makedirs('debug/',exist_ok=True)

def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),  # 학습 시에도 이런 식으로 썼다면 맞춰준다
        ToTensor()                           
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/',type=str, help='the path of dataset')
    parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs(f'debug/',exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_files = list(df.image_id)
    test_dataset = multi_classes(args.dataset,test_files, get_transform())
    model = torch.load(args.model)
    model = model.cuda()
    sfx = nn.Softmax(dim=1)
    # cfs = ConfusionMatrix(3)
    cfs = ConfusionMatrix(4)
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    since = time.time()
    
    for image_id in test_files:
        # print(image_id)
        img = cv2.imread(f'/userHome/userhome4/kyoungmin/dataset/data/{image_id}/{image_id}.png')
        # img = cv2.imread(f'data/{image_id}/images/{image_id}.png')
        img = cv2.resize(img, ((240,240)))
        cv2.imwrite(f'debug/{image_id}.png',img)

    with torch.no_grad():
        # for img, mask, mask2, img_id in test_dataset:
        for img, mask, img_id in test_dataset:
        
            # 채널 별로 찍먹하기
            # print(mask.shape)
            # mask_np = mask.cpu().numpy()  # (H, W) 형태의 numpy 배열
            # for c in range(mask_np.shape[0]):
            #     # c번째 채널은 shape (256, 256)의 2D 배열
            #     class_slice = mask_np[c]  # [H, W]

            #     # 채널 값이 1(혹은 특정 임계치 이상)이면 255, 아니면 0
            #     class_mask = (class_slice > 0.5).astype(np.uint8) * 255

            #     cv2.imwrite(f'debug/{img_id}_c{c}.png', class_mask)

            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)

            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)
            
            ts_sfx = sfx(pred)
            pred = sfx(pred)
            
            # print('pred',pred.min(), pred.max())
            # print('mask',mask.min(), mask.max())
            # cv2.imwrite(f'debug/mask.png',mask.cpu().numpy().astype('uint8'))
            # cv2.imwrite(f'debug/pred.png',pred.cpu().numpy().astype('uint8'))

            img_class = torch.max(ts_sfx,1).indices.cpu()
            pred = torch.max(pred,1).indices.cpu()
            mask = torch.max(mask,1).indices.cpu()
            mask_draw = mask.clone().detach()

            # print('aftpred',pred.min(), pred.max())
            # print('aftmask',mask.min(), mask.max())
            
            if args.debug:
        
                img_numpy = pred.detach().numpy()[0]
                img_numpy[img_numpy==1] = 63
                img_numpy[img_numpy==2] = 127
                img_numpy[img_numpy==3] = 255
                cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy)
                # print(img_id)
                
                mask_numpy = mask_draw.detach().numpy()[0]
                mask_numpy[mask_numpy==1] = 63
                mask_numpy[mask_numpy==2] = 127
                mask_numpy[mask_numpy==3] = 255
                # print(mask_numpy)
                cv2.imwrite(f'debug/{img_id}_gt.png',mask_numpy)
               
            cfsmat = cfs(img_class,mask).numpy()
            
            sum_iou = 0
            sum_prec = 0
            sum_acc = 0
            sum_recall = 0
            sum_f1 = 0
            sum_dice = 0.0

            ###
            # 클래스 갯수 3->4로 변환환
            # for i in range(3):
            for i in range(4):
                tp = cfsmat[i,i]
                fp = np.sum(cfsmat[0:4,i]) - tp
                fn = np.sum(cfsmat[i,0:4]) - tp
                # fp = np.sum(cfsmat[0:,i]) - tp
                # fn = np.sum(cfsmat[i,0:]) - tp
                # fp = np.sum(cfsmat[0:3,i]) - tp
                # fn = np.sum(cfsmat[i,0:3]) - tp
              
                # 다중 클래스라 0이 나오는 case 존재
                tmp_iou = tp / (fp + fn + tp)
                tmp_prec = tp / (fp + tp + 1) 
                tmp_acc = tp
                tmp_recall = tp / (tp + fn)
                tmp_dice = 2 * tp / (2 * tp + fp + fn )
                
                sum_prec += tmp_prec
                sum_acc += tmp_acc
                
                # 혹시 모를 NaN 방지
                if math.isnan(tmp_dice):
                    pass
                else:
                    sum_dice += tmp_dice
                if math.isnan(tmp_iou):
                    pass
                else:
                    sum_iou += tmp_iou
                if math.isnan(tmp_recall):
                    pass
                else:
                    sum_recall += tmp_recall
                
            
            sum_acc /= (np.sum(cfsmat)) 
            sum_prec /= 4
            sum_recall /= 4
            sum_iou /= 4
            sum_dice /= 4
            sum_f1 = 2 * sum_prec * sum_recall / (sum_prec + sum_recall)
            
            iou_score.append(sum_iou)
            acc_score.append(sum_acc)
            pre_score.append(sum_prec)
            recall_score.append(sum_recall)
            dice_score.append(sum_dice)

            if math.isnan(sum_recall): 
                print(img_id)
            # print(f'sum_recall : {sum_recall}')
            f1_score.append(sum_f1)
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',np.mean(iou_score),np.std(iou_score))
    print('mean accuracy:',np.mean(acc_score),np.std(acc_score))
    print('mean precsion:',np.mean(pre_score),np.std(pre_score))
    print('mean recall:',np.mean(recall_score),np.std(recall_score))
    print('mean F1-score:',np.mean(f1_score),np.std(f1_score))
    print('mean Dice:', np.mean(dice_score), np.std(dice_score))
