import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinv
from numpy import fft
import math
import cv2


def get_motion_dsf(image_size, motion_dis, motion_angle):
    PSF = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2
    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return PSF / PSF.sum()


def make_blurred(input, PSF):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF)
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))

    return blurred


def add_gaussian_noise(input, sigma):
    if sigma < 0:
        return input
    gauss = np.random.normal(0, sigma, np.shape(input))
    noisy_img = input + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255

    return noisy_img


def wiener_filter_1(input, kernel, K=0.02):
    img_fft = fft.fft2(input)
    kernel_fft = fft.fft2(kernel)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.fft.ifft2(img_fft * kernel_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result


def wiener_filter_2(input, noisy, origin, kernel):
    img_fft = fft.fft2(input)
    kernel_fft = fft.fft2(kernel)
    noisy_fft = fft.fft2(noisy)
    origin_fft = fft.fft2(origin)
    NSR = (np.abs(noisy_fft) ** 2)/(np.abs(origin_fft) ** 2)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + NSR)
    result = fft.ifft2(img_fft * kernel_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result


if __name__ == "__main__":
    img = cv2.imread('default.jpg', 0)
    PSF = get_motion_dsf(img.shape, 40, 45)
    blurred = make_blurred(img, PSF)
    blur_noisy = add_gaussian_noise(blurred, 25)
    noisy = blur_noisy-blurred
    result_1 = wiener_filter_1(blur_noisy, PSF)
    result_2 = wiener_filter_2(blur_noisy, noisy, img, PSF)
    plt.subplot(221)
    plt.title('Origin')
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, 'gray')
    plt.subplot(222)
    plt.title('Blurred+Gauss')
    plt.xticks([]), plt.yticks([])
    plt.imshow(blur_noisy, 'gray')
    plt.subplot(223)
    plt.title('Unknown NSR')
    plt.xticks([]), plt.yticks([])
    plt.imshow(result_1, 'gray')
    plt.subplot(224)
    plt.title('Known NSR')
    plt.xticks([]), plt.yticks([])
    plt.imshow(result_2, 'gray')
    plt.show()

