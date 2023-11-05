import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__=='__main__':
   loc_000_sharp = 'C:/AG/course notes/111_2/deblur final project/dataset download/train_sharp_bicubic/train/train_sharp_bicubic/X4/000/00000014.png'
   loc_000_blur = 'C:/AG/course notes/111_2/deblur final project/dataset download/train_blur_bicubic/train/train_blur_bicubic/X4/000/00000014.png'
   loc_000_blurdeblur = 'C:/AG/course notes/111_2/deblur final project/image for report/fromBlur_00000014_x1_usrnet_tiny.png'
   loc_000_sharpdeblur = 'C:/AG/course notes/111_2/deblur final project/image for report/fromSharp_00000014_x1_usrnet_tiny.png'

   img_000_sharp = cv2.imread(loc_000_sharp)
   img_000_blur = cv2.imread(loc_000_blur)
   img_000_blurdeblur = cv2.imread(loc_000_blurdeblur)
   img_000_sharpdeblur = cv2.imread(loc_000_sharpdeblur)

   print('000_blur_psnr:', psnr(img_000_sharp, img_000_blur))
   print('000_blurdeblur_psnr:', psnr(img_000_sharp, img_000_blurdeblur))
   print('000_sharpdeblur_psnr:', psnr(img_000_sharp, img_000_sharpdeblur))   # 沒意義


   print(' ')


   loc_014_sharp = 'C:/AG/course notes/111_2/deblur final project/dataset download/train_sharp_bicubic/train/train_sharp_bicubic/X4/014/00000017.png'
   loc_014_blur = 'C:/AG/course notes/111_2/deblur final project/dataset download/train_blur_bicubic/train/train_blur_bicubic/X4/014/00000017.png'
   loc_014_blurdeblur = 'C:/AG/course notes/111_2/deblur final project/image for report/fromBlur_00000017_x1_usrnet_tiny.png'
   loc_014_sharpdeblur = 'C:/AG/course notes/111_2/deblur final project/image for report/fromSharp_00000017_x1_usrnet_tiny.png'

   img_014_sharp = cv2.imread(loc_014_sharp)
   img_014_blur = cv2.imread(loc_014_blur)
   img_014_blurdeblur = cv2.imread(loc_014_blurdeblur)
   img_014_sharpdeblur = cv2.imread(loc_014_sharpdeblur)

   print('017_blur_psnr:', psnr(img_014_sharp, img_014_blur))
   print('017_blurdeblur_psnr:', psnr(img_014_sharp, img_014_blurdeblur))
   print('017_sharpdeblur_psnr:', psnr(img_014_sharp, img_014_sharpdeblur))