import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1/255.0 - img2/255.0) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def DPCM(yBuffer, dBuffer, re, w, h, bitnum):
    x = 2 ** (8 - bitnum)
    y = 2 ** (9 - bitnum)
    flow_upper_bound = 2 ** bitnum - 1
    for i in range(0, h):
        prediction = 128
        pred_error = yBuffer[i * w] - prediction
        tmp = (pred_error + 128) // x
        dBuffer[i * w] = tmp
        inv_pred_error = dBuffer[i * w] * x - 128
        re[i * w] = inv_pred_error + prediction
        for j in range(1, w):
            prediction = re[i * w + j - 1]
            predErr = yBuffer[i * w + j] - prediction
            tmp = (predErr + 255) // y
            dBuffer[i * w + j] = tmp
            invPredErr = dBuffer[i * w + j] * y - 255
            re[i * w + j] = invPredErr + prediction


Img = cv2.imread('default.jpg')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
h = Img.shape[0]
w = Img.shape[1]
PSNR = []
SSIM = []

# 1 bit 重构
dBuffer1 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer1 = np.zeros((h * w), dtype=np.uint8)
yBuffer1 = Img.reshape(h * w)
DPCM(yBuffer1, dBuffer1, rebuildBuffer1, w, h, 1)
dBuffer1 = dBuffer1.reshape(h, w)
rebuildBuffer1 = rebuildBuffer1.reshape(h, w)
PSNR.append(psnr(Img, rebuildBuffer1))
SSIM.append(ssim(Img, rebuildBuffer1, multichannel=True))

# 2 bit 重构
dBuffer2 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer2 = np.zeros((h * w), dtype=np.uint8)
yBuffer2 = Img.reshape(h * w)
DPCM(yBuffer2, dBuffer2, rebuildBuffer2, w, h, 2)
dBuffer2 = dBuffer2.reshape(h, w)
rebuildBuffer2 = rebuildBuffer2.reshape(h, w)
PSNR.append(psnr(Img, rebuildBuffer2))
SSIM.append(ssim(Img, rebuildBuffer2, multichannel=True))

# 4 bit 重构
dBuffer4 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer4 = np.zeros((h * w), dtype=np.uint8)
yBuffer4 = Img.reshape(h * w)
DPCM(yBuffer4, dBuffer4, rebuildBuffer4, w, h, 4)
dBuffer4 = dBuffer4.reshape(h, w)
rebuildBuffer4 = rebuildBuffer4.reshape(h, w)
PSNR.append(psnr(Img, rebuildBuffer4))
SSIM.append(ssim(Img, rebuildBuffer4, multichannel=True))

# 8 bit 重构
dBuffer8 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer8 = np.zeros((h * w), dtype=np.uint8)
yBuffer8 = Img.reshape(h * w)
DPCM(yBuffer8, dBuffer8, rebuildBuffer8, w, h, 8)
dBuffer8 = dBuffer8.reshape(h, w)
rebuildBuffer8 = rebuildBuffer8.reshape(h, w)
PSNR.append(psnr(Img, rebuildBuffer8))
SSIM.append(ssim(Img, rebuildBuffer8, multichannel=True))

print('PSNR:', PSNR)
print('SSIM:', SSIM)

plt.subplot(221)
plt.title('1-bit')
plt.xticks([]), plt.yticks([])
plt.imshow(rebuildBuffer1, 'gray')
plt.subplot(222)
plt.title('2-bit')
plt.xticks([]), plt.yticks([])
plt.imshow(rebuildBuffer2, 'gray')
plt.subplot(223)
plt.title('4-bit')
plt.xticks([]), plt.yticks([])
plt.imshow(rebuildBuffer4, 'gray')
plt.subplot(224)
plt.title('8-bit')
plt.xticks([]), plt.yticks([])
plt.imshow(rebuildBuffer8, 'gray')
plt.show()
