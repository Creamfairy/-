import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('default.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
row, col = img.shape[0:2]

# 图像平移
M = np.float32([[1, 0, 150], [0, 1, 100]])
result1 = cv2.warpAffine(img, M, (col, row))

# 图像镜像
result2 = cv2.flip(img, 1)

# 图像旋转
M = cv2.getRotationMatrix2D((col/2, row/2), 30, 1)
result3 = cv2.warpAffine(img, M, (col, row))

# 图像平移，镜像，旋转复合
step1 = cv2.flip(img, 1)
M1 = cv2.getRotationMatrix2D((col/2, row/2), 30, 1)
step2 = cv2.warpAffine(step1, M1, (col, row))
M2 = np.float32([[1, 0, 150], [0, 1, 100]])
result4 = cv2.warpAffine(step2, M2, (col, row))

# 图像显示
plt.figure(num='comparison')
titles = ['origin', 'Translation', 'mirroring', 'rotation', 'composite']
images = [img, result1, result2, result3, result4]
for i in range(len(images)):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
