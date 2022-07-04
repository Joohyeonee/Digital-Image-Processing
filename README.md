# Digital-Image-Processing(Spatial Filtering) 
 - 영상보다 작은 커널을 정의하고 해당 영역 안에서 연산을 수행한 뒤 다음 픽셀로 옮겨서 진행
 - 이미지를 구성하는 특정한 frequency를 변형시키는 것
 - 각각의 픽셀값들을 이웃한 픽셀값들의 함수에 의해서 바꾼다

2. linear average filter : 불필요한 노이즈 제거 효과
 - 이웃하는 픽셀과 combination을 이루어 값을 바꿀 수 있다
 - 필터의 평균값을 구한 후 Input과 Sum of Product 하는 방식

 - convolution, cross-correlation
 >>(f * g)(x, y) = sum f(i, j)I(x-i, y-j) / (f x g)(x, y) = sum f(i, j)I(x+i, y+j)
 >> 원리
 >>gathering : 여러 픽셀 정보로 하나의 픽셀값 결정
 >>scattering : 하나의 픽셀 정보로 여러 픽셀값 결정

 - gaussian filter
 >>디테일한 부분을 없애고 smooth한 이미지 만들 시
 >>filter의 중심값이 가장 크고 주변부로 갈 수록 값이 줄어듦
 >>blurry하게 만드는 효과

2. sharpening filter : 엣지를 얻어내야함
 - 2차 미분 이용(앞-뒤 데이터만 아니라 이전, 현재, 이후 데이터 정보를 함께 사용) : Laplacian 미분 적용 f(x+1, y) + f(x-1, y)+ f(x, y+1) + f(x, y-1) - 4f(x, y)
 - unsharp masking : original image - averaged image 방식
 - ramp : 영상 내에서 픽셀값이 완만하게 움직이는 구간
 - step : 영상 내에서 픽셀값이 급격하게 움직이는 구간
 - 2차 미분 수행 시 step 지점에서 zero crossing 발생 -> 엣지를 찾을 수 있음
 >>등방성 필터 : 회전 불변성 필터(라플라시안 필터)
 >>비등방성 필터 : 회전 가변성 필터(그래디언트 이용-로버츠 교차-그래디언트 연산자)
 
3. Median filter
 - 중앙값을 선택해서 픽셀값 산출(계산량이 큰 sorting 필요)

4. smoothing filter
 - blurring, noise 제거를 위해 사용하는 방법(noise는 주변 픽셀값과 큰 차이를 보이므로 평준화를 통한 제거 가능)
 - linear와 non-linear로 나뉨
 - smoothing linear filter(average filter)
 - order-statistic filter
 >>kernel 내에서 중앙값을 Max or Median or Min으로 교체하는 방식(흑/백 점들이 흩어진 영상에 효과적)

>Cross-Correlation vs Convolution
>유사성의 비교 vs 다른 signal 대비 이 signal 비교
