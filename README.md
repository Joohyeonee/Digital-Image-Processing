# Digital-Image-Processing(Spatial Filtering)
 - 이미지를 구성하는 특정한 frequency를 변형시키는 것
 - 각각의 픽셀값들을 이웃한 픽셀값들의 함수에 의해서 바꾼다

2. linear average filter : 불필요한 노이즈 제거 효과
 - 이웃하는 픽셀과 combination을 이루어 값을 바꿀 수 있다
 - 필터의 평균값을 구한 후 Input과 Sum of Product 하는 방식

 - convolution, cross-correlation
 >(f * g)(x, y) = sum f(i, j)I(x-i, y-j) / (f x g)(x, y) = sum f(i, j)I(x+i, y+j)
 > 원리
 >gathering : 여러 픽셀 정보로 하나의 픽셀값 결정
 >scattering : 하나의 픽셀 정보로 여러 픽셀값 결정

 - gaussian filter
 >디테일한 부분을 없애고 smooth한 이미지 만들 시
 >filter의 중심값이 가장 크고 주변부로 갈 수록 값이 줄어듦
 >blurry하게 만드는 효과

2. sharpening filter
 - 미분 이용 : Laplacian 미분 적용[f(x+1, y) + f(x-1, y)+ f(x, y+1) + f(x, y-1) - 4f(x, y)]
 - unsharp masking : original image - averaged image 방식
 
3. Median filter
 - 중앙값을 선택해서 픽셀값 산출(계산량이 큰 sorting 필요)

>Cross-Correlation vs Convolution
>유사성의 비교 vs 다른 signal 대비 이 signal 비교







