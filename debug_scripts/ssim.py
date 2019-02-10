import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("/home/tk/Documents/recover/MSEloss/0_100.png")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1, requires_grad = False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).data.item()
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.90:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.data.item()
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()


img2 = img2[0][0].view(128,128).detach().numpy() * 255
cv2.imwrite("/home/tk/Documents/test.jpg", img2)
