import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
# from albumentations.pytorch import ToTensor

# def Normalization():
#    return A.Compose(
#        [
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensor()
#         ])
def Normalization():
    return A.Compose(
        [
            A.Normalize(mean=(0.5, ), std=(0.5, )),  # Grayscale 평균과 표준편차
            # ToTensor()
        ])

#Dataset Loader
class multi_classes(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
            self.normalization = Normalization()
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            # image_folder = os.path.join(self.path,str(self.folders[idx]),'images/')
            image_folder = os.path.join(self.path, str(self.folders[idx]))
            # print(f"image_folder : {image_folder}")
            # print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
            # mask_folder = os.path.join(self.path,str(self.folders[idx]),'masks/')
            # image_folder = os.path.join(self.path,str(self.folders[idx]))
            # image_folder = os.path.join(self.path,'images/')
            # mask_folder = os.path.join(self.path,'masks/')
            folder_name = os.path.basename(image_folder)
            image_path = os.path.join(image_folder,f"{folder_name}.png")
            mask_path = os.path.join(image_folder,f"{folder_name}_mask.png")
            # mask_path = os.path.join(image_folder,os.listdir(image_folder)[0] + "_mask.png")
            # print(f"image path : {image_path}") 
            # print(f"mask path : {mask_path}")
            # mask_path = os.path.join(mask_folder,os.listdir(mask_folder)[0])
            image_id = self.folders[idx]
            img = io.imread(image_path)
            if img.ndim == 2:  # 이미지가 2차원(Grayscale)인 경우
                img = np.expand_dims(img, axis=-1).astype('float32')  # 1채널로 유지
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'tmp/{idx}_mask.png',mask)
            mask = cv2.resize(mask, (240, 240), interpolation=cv2.INTER_NEAREST)
            # else:  # 이미지가 이미 3채널인 경우
            #     img = img[:, :, 0:1].astype('float32')  # 첫 번째 채널만 사용
            #     print('color')
            # img = io.imread(image_path).astype('float32')     
            # mask = self.get_mask(mask_folder, 240, 240)
            # mask = io.imread(mask_path).astype('float32')     
            # print(img.dtype, mask.dtype)
            # print(img.shape, mask.shape)
            # 타입과 차원 확인

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            # normalized = self.normalization(image=img, mask=mask)
            # img_nl = normalized['image']
            # mask_nl = normalized['mask']
            # mask_nl = np.squeeze(mask_nl)
            
            # mask = img_as_ubyte(mask) 
            # mask = np.squeeze(mask)
            # mask[(mask > 0) & (mask < mask.max())] = 1
            # mask[mask == mask.max()] = 2
            # mask = torch.from_numpy(mask)
            # mask = torch.squeeze(mask)
            # mask = torch.nn.functional.one_hot(mask.to(torch.int64),3)
            # mask = mask.permute(2, 0, 1)

            mask = img_as_ubyte(mask)
            mask = np.squeeze(mask)

            # 4개의 클래스를 one-hot 인코딩하기 위해 4를 3으로 변환
            mask[mask == 1] = 1  # 클래스 1
            mask[mask == 2] = 2  # 클래스 2
            mask[mask == 4] = 3  # 클래스 4 (3으로 변환)
            
            # mask_numpy = mask[0]
            # mask_numpy[mask_numpy==1] = 63
            # mask_numpy[mask_numpy==2] = 127
            # mask_numpy[mask_numpy==3] = 255
            # cv2.imwrite(f'debug/{idx}_gt_v2.png',mask_numpy)

            # Torch tensor로 변환 및 one-hot 인코딩
            # mask = torch.from_numpy(mask).to(torch.int64)
            mask = torch.from_numpy(mask).to(torch.int64)
            mask = torch.nn.functional.one_hot(mask, num_classes=4)  # num_classes=4로 변경
            # print(mask.shape)
            mask = mask.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            # print(mask.shape)
            # return (img_nl,mask,mask_nl,image_id) 

            return (img,mask,image_id) 


        # def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):
        #     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)  # 초기 마스크

        #     for mask_file in os.listdir(mask_folder):
        #         # 마스크 파일 읽기 및 리사이즈
        #         mask_path = os.path.join(mask_folder, mask_file)
        #         mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 바로 그레이스케일로 읽기
        #         mask_ = cv2.resize(mask_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

        #         # inplace로 최대값 갱신
        #         np.maximum(mask, mask_, out=mask)

        #     return mask

class binary_class(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            # image_path = os.path.join(self.path, self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])
            
            # img = io.imread(image_path)
            rgb_img = io.imread(image_path).astype('float32')

                # 흑백 이미지인지 확인 후 처리
            if len(rgb_img.shape) == 3:  # 3채널(BGR) 이미지일 경우
                img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            elif len(rgb_img.shape) == 2:  # 이미 흑백 이미지
                img = rgb_img
            else:
                raise ValueError(f"Unexpected image shape: {rgb_img.shape}")
            # print(img)
            # img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

            # img = io.imread(image_path)[:,:,:3].astype('float32')
            mask = io.imread(mask_path)
            # print(mask.shape)
            image_id = self.folders[idx]
            # print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
            # print(f"Image shape: {img.shape}, dtype: {img.dtype}")
            # print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            return (img,mask,image_id)
        
class binary_class2(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,self.folders[idx],'images/',self.folders[idx])
            # image_path = os.path.join(self.path,self.folders[idx], self.folders[idx])
            mask_path = os.path.join(self.path,self.folders[idx],'masks/',self.folders[idx])
            image_id = self.folders[idx]
            img = io.imread(f'{image_path}.png')[:,:,:3].astype('float16')
            mask = io.imread(f'{mask_path}.png', as_gray=True)

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
   
            return (img,mask,image_id)
        
