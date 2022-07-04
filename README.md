# Digital-Image-Processing(Spatial Filtering)
1. spatial filter
 - 이미지를 구성하는 특정한 frequency를 변형시키는 것
 - 각각의 픽셀값들을 이웃한 픽셀값들의 함수에 의해서 바꾼다

2. linear filter
 - 이웃하는 픽셀과 combination을 이루어 값을 바꿀 수 있다
 - 필터의 평균값을 구한 후 Input의 해당하는 각 요소에 곱하여 합을 구하는 방식

3. convolution
 - (f * g)(x, y) = sum f(i, j)I(x-i, y-j)
 - gathering : 여러 픽셀 정보로 하나의 픽셀값 결정
 - scattering : 하나의 픽셀 정보로 여러 픽셀값 결정

4. gaussian filter
 - 디테일한 부분을 없애고 smooth한 이미지 만들 시
 - filter의 중심값이 가장 크고 주변부로 갈 수록 값이 줄어듦
 - blurry하게 만드는 효과

5. sharpening filter
 - 미분 이용 : Laplacian 미분 적용[f(x+1, y) + f(x-1, y)+ f(x, y+1) + f(x, y-1) - 4f(x, y)]
 - unsharp masking : original image - averaged image 방식
 
6. Median filter
 - 중앙값을 선택해서 픽셀값 산출(계산량이 큰 sorting 필요)




