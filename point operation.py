import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('default.jpg', 0)

# 线性点运算
a = 1.5
b = 0
l_img = a * img + b
l_img = np.uint8(l_img*255/np.max(l_img))

# 分段点运算
M = 255
m = 200
a = 80
b = 180
c = 40
d = 160
d_img = np.zeros_like(img)
for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        if img[h][w] < a:
            d_img[h][w] = (c/a)*img[h][w]
        elif a <= img[h][w] < b:
            d_img[h][w] = ((d-c)/(b-a))*(img[h][w]-a)+c
        elif img[h][w] >= b:
            d_img[h][w] = ((m-d)/(M-b))*(img[h][w]-b)+d

# 非线性点运算
c = 1
v = 4.0
nl_img = c*np.power(img, v)
nl_img = np.uint8(nl_img*255/np.max(nl_img))


plt.figure(num='comparison')
titles = ['origin', 'linear', 'segmentation', 'non-linear']
images = [img, l_img, d_img, nl_img]
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


