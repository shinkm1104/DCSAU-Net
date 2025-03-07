import cv2
import numpy as np
mask_numpy = cv2.imread(r'/userHome/userhome4/kyoungmin/code/DCSAU-Net/tmp/75_mask.png',0)
mask_numpy[mask_numpy==1] = 63
mask_numpy[mask_numpy==2] = 127
mask_numpy[mask_numpy==4] = 255
cv2.imwrite('tmp.png', mask_numpy)
print(np.unique(mask_numpy))
