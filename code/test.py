import cv2
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
from torchinfo import summary

import tensorflow as tf

# import tf_USRNet
import original_USRNet as USRNet

# 讀資料
img = cv2.imread('C:/AG/course notes/111_2/deblur final project/dataset download/val_blur/val/val_blur/000/00000084.png')
data = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

# 載入預訓練模型
pth = torch.load("C:/AG/course notes/111_2/deblur final project/pre_train/usrnet.pth")
model = USRNet.USRNet()
model.load_state_dict(pth)

# 預測資料+儲存
kernels = loadmat('kernels_12.mat')['kernels']
kernel = kernels[0, 0].astype(np.float64)
k = torch.from_numpy(np.ascontiguousarray(kernel[..., np.newaxis])).permute(2, 0, 1).float().unsqueeze(0)

noise_level_img = 0
noise_level_model = noise_level_img
sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])

predict = model(data,k,1,sigma)
numpy_array = predict.cpu().detach().numpy()
print(predict.type)
cv2.imwrite('output_001.png', predict)