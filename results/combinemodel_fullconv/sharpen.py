
import cv2
import numpy as np
import torch

img_np = cv2.imread("0_tar.png")[:,:,0]
img = torch.Tensor( np.array([[img_np]]) )
matrix = torch.Tensor(np.array([[0,1,0],[1,-4,1],[0,1,0]]).reshape(1,1,3,3))

conv = np.absolute( torch.nn.functional.conv2d(img, matrix) ).detach().numpy().squeeze()

cv2.imwrite("0_sharpen.png", conv + img_np[1:-1,1:-1])