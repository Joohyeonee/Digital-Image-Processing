import cv2
kernel = cv2.getGaussianKernel(5, 3) #kernel size = 5, sigma = 3
print(kernel)
import math
import numpy as np
kernel1d = cv2.getGaussianKernel(5, 3)
kernel2d = np.outer(kernel1d, kernel1d.transpose())
print(kernel2d)

#Apply Gaussian Filter
from PIL import Image
im = Image.open('newskin.jpg') # Image load
im_array = np.asarray(im) # Image to np.array

kernel1d = cv2.getGaussianKernel(5, 3)
kernel2d = np.outer(kernel1d, kernel1d.transpose())

low_im_array = cv2.filter2D(im_array, -1, kernel2d) # convolve

low_im = Image.fromarray(low_im_array) # np.array to Image
low_im.save('low_newskin.bmp','BMP')

high_im_array = im_array - low_im_array + 128
high_im = Image.fromarray(high_im_array)
high_im.save('high_newskin.bmp','BMP')


#Gaussian Filter 생성
def get_gaussian_filter_1d(size, sigma):
    """
    1D 가우시안 필터를 생성한다.
    :param size: int 커널 사이즈
    :param sigma: float
    :return kernel: np.array
    """
    assert size % 2 == 1, "Filter Dimension must be odd" # filter size는 무조건 odd여야한다
    arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2)+1 ,1) # 중심으로 부터의 거리가 값인 배열 생성
    kernel_raw = np.exp((-arr*arr)/(2*sigma*sigma)) # 가우시안 필터 공식
    kernel = kernel_raw/kernel_raw.sum() # 정규화
    return kernel

print(get_gaussian_filter_1d(5, 3))