# Digital-Image-Processing
Spatial Filtering
- 영상보다 작은 커널을 정의하고 해당 영역 안에서 연산을 수행한 뒤 다음 픽셀로 옮겨서 진행
- 이미지를 구성하는 특정한 frequency를 변형시키는 것
- 각각의 픽셀값들을 이웃한 픽셀값들의 함수에 의해서 바꾼다

1. linear average filter : 불필요한 노이즈 제거 효과
- 필터의 평균값을 구한 후 Input과 Sum of Product 하는 방식
- convolution, cross-correlation
>(f * g)(x, y) = sum f(i, j)I(x-i, y-j) / (f x g)(x, y) = sum f(i, j)I(x+i, y+j)
> 원리
>>gathering : 여러 픽셀 정보로 하나의 픽셀값 결정
>>scattering : 하나의 픽셀 정보로 여러 픽셀값 결정

2. sharpening filter : 엣지를 얻어내야함
- 2차 미분 이용(앞-뒤 데이터만 아니라 이전, 현재, 이후 데이터 정보를 함께 사용)
- unsharp masking : original image - averaged image 방식
- ramp : 영상 내에서 픽셀값이 완만하게 움직이는 구간
- step : 영상 내에서 픽셀값이 급격하게 움직이는 구간
- 2차 미분 수행 시 step 지점에서 zero crossing 발생 -> 엣지를 찾을 수 있음
>등방성 필터 : 회전 불변성 필터(라플라시안 필터)
>>Laplacian Filter
>> - f(x+1, y) + f(x-1, y)+ f(x, y+1) + f(x, y-1) - 4f(x, y)
>> - g(x, y) = f(x, y) + c[delta(f(x, y))] / g(x, y) : 샤프닝 된 영상, f(x, y) : 샤프닝 전 영상
```
#Laplacian Filter
import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
img0 = cv2.imread('SanFrancisco.jpg',)
#img0 = cv2.imread('windows.jpg',)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
```
>언샤프 마스킹, 하이부스트 필터링 : 입력 영상 블러링 -> 마스크 얻음(입력 영상 - 블러링 된 영상) -> 마스크를 입력 영상에 더함
>>g_mask(x, y) = f(x, y) + k * g_mask(x, y)
>> k = 1 : 언샤프 마스킹, k > 1 : 하이부스트 필터링
>비등방성 필터 : 회전 가변성 필터(그래디언트 이용-로버츠 교차-그래디언트 연산자)
>>Roberts Cross Filter
>> - Convolution kernel : 대각선 방향으로 +1과 -1을 배치시켜 검출 효과를 높임, 노이즈에 민감
```
import cv2 
import numpy as np
from scipy import ndimage
  
roberts_cross_v = np.array( [[1, 0 ],
                             [0,-1 ]] )
  
roberts_cross_h = np.array( [[ 0, 1 ],
                             [ -1, 0 ]] )
  
img = cv2.imread("input.webp",0).astype('float64')
img/=255.0
vertical = ndimage.convolve( img, roberts_cross_v )
horizontal = ndimage.convolve( img, roberts_cross_h )
  
edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255
cv2.imwrite("output.jpg",edged_img)
```

>>Prewitt Filter
>> - x축과 y축의 방향으로 차분을 3번 계산하여 경계 검출, 상하-좌우 경계는 잘 검출하지만 대각선 검출이 약함
>>Sobel Filter
>> - 중심 픽셀의 차분 비중을 2배로 준 필터(현재 많이 쓰여 OpenCV에서 별도의 함수 제공)
>> dst = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
 
3. Median filter
- 중앙값을 선택해서 픽셀값 산출(계산량이 큰 sorting 필요)

4. smoothing filter
- blurring, noise 제거를 위해 사용하는 방법(noise는 주변 픽셀값과 큰 차이를 보이므로 평준화를 통한 제거 가능)
- linear와 non-linear로 나뉨
- smoothing linear filter(average filter)
- order-statistic filter
>kernel 내에서 중앙값을 Max or Median or Min으로 교체하는 방식(흑/백 점들이 흩어진 영상에 효과적)
- gaussian filter(Linear filter)
>디테일한 부분을 없애고 smooth한 이미지 만들 시
>filter의 중심값이 가장 크고 주변부로 갈 수록 값이 줄어듦
>blurry하게 만드는 효과
```
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
    assert size%2==1, "Filter Dimension must be odd" # filter size는 무조건 odd여야한다
    arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2)+1 ,1) # 중심으로 부터의 거리가 값인 배열 생성
    kernel_raw = np.exp((-arr*arr)/(2*sigma*sigma)) # 가우시안 필터 공식
    kernel = kernel_raw/kernel_raw.sum() # 정규화
    return kernel

print(get_gaussian_filter_1d(5, 3))
```
>Cross-Correlation vs Convolution : 유사성의 비교 vs 다른 signal 대비 이 signal 비교

Frequency Domain Filtering
1. Fourier Series
- 어떤 함수 f(t)가 주기(T)와 연속형 변수(t)를 가질 때,  f(t) = a * sin(t) + b * cos(t)로 표현될 수 있음 / sin과 cos의 linear combination으로 표현 가능
2. Impulse
- delta(t) = infinite(t = 0), 0(t != 0) (t:continuous)
- delta(x) = 1(x = 0), 0(x != 0) (x:discrete)
3. Fourier Transform
- 이미지를 주파수 영역으로 전환하여 이미지 프로세싱 작업이 가능하도록 하는 도구
- x 또는 y축을 시간축으로 놓고 좌표의 변화에 따라 변하는 이미지 픽셀의 밝기 변화를 신호로 생각
- W x H 이미지에 대한 이산 푸리에 변환에서 F(u, v)는 u, v성분이 아니라 u/W, v/H 성분에 대한 계수를 나타냄
4. Convolution Theorem
- 영상 공간에서의 convolution 연산 = 주파수 공간에서 두 함수의 푸리에 변환의 단순 곱 : f(t) * h(t) = F(m)H(m)
- 주파수 공간에서의 convolution 연산 = 영상 공간에서 두 함수의 역푸리에 변환의 단순 곱 : f(t)h(t) = F(m) * H(m)
5. 주파수 공간에서의 영상 필터링 단계
>1. 영상에 이산 푸리에 변환 F를 적용하여 영상 공간 -> 주파수 공간으로 변환
>2. 변환 함수를 사용하여 필터링 수행
>3. 이산 역푸리에 변환 F-1을 적용하여 주파수 공간 -> 영상 공간으로 변환
>수학적 모델링 식
>>g(x, y) = F-1(H(m, v)F(m, v)) / F(m, v) : 입력 영상에 DFT 적용, H(m, v) : 필터링 함수
6. 주파수 공간의 특성
- 영상 내 저주파 영역 : 변화가 작은 곳(구름 없는 하늘 등) -> 저역 통과 필터링
- 영상 내 고주파 영역 : 변화가 많은 곳(엣지, 노이즈 등 밝기의 변화가 급격한 곳) -> 고역 통과 필터링
- 원점에 대한 대칭성 존재 -> 특정 영역의 원점 대칭인 부분까지 필터링 수행
7. 주파수 도메인 필터를 이용한 영상 스무딩
 - ILPF(이상적 저역통과 필터) : 중앙을 기준으로 특정 반지름 D0 안에 있으면 1, 아니면 0으로 만들어주는 필터 / 이상적 필터 : 특정 주파수를 0으로 만듦
 - BLPF(Butterworth 저역통과 필터) : 0과 1 사이에서 포물선을 그리기 때문에 ILPF와 빅하면 훨씬 부드러움
 - GLPF(Gaussian 저역통과 필터) : ILPF, BLPF에 비해 훨씬 부드러운 필터(부드러운 스무딩)
8. 주파수 도메인 필터를 이용한 영상 샤프닝 : 고주파 영역인 edge를 가져옴
 - IHPF
 - BHPF
 - GHPF
 - Laplacian : 공간 도메인에서와 같은 효과
 - 준동형필터링 : 밝기 범위 압축 및 대비 향상을 동시에 하는 필터링 기법
9. 주파수 도메인 필터링에 의한 노이즈 감소
 - 선택적 필터링
 >대역차단 필터 H_BP(m, v) : 특정 주파수 영역을 차단하는 필터
 >대역통과 필터 H_BR(m, v) : 1 - H_BP(m, v)
 - 노치 필터 : 선택적 필터링 중 사용자의 개입이 가장 많이 들어가는 필터링 : 중심이 노치의 중심으로 이동된 고역 통과 필처들의 곱으로 만들어짐
 - 최적 노치 필터링 : 노치 필터링을 통해 복원된 영상 f(x, y)의 지역적 분산 최소화
 >간섭 패턴의 주요 주파수 성분을 빼는 것
