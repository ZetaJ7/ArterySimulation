import cv2


cv2.namedWindow('canny')
cv2.createTrackbar('maxVal', 'canny', 0, 255, nothing)
cv2.createTrackbar('minVal', 'canny', 0, 255, nothing)

img = cv2.imread('1.jpg', 0)

while cv2.waitKey(1) != 27:
    maxVal = cv2.getTrackbarPos('maxVal', 'canny')
    minVal = cv2.getTrackbarPos('minVal', 'canny')

    #    第一步是使用 5x5 的高斯滤波器去除噪声
    #    第二步计算图像梯度：sobel
    #    第三步非极大值抑制：
    #    对整幅图像做一个扫描，去除那些非边界上的点
    #    对每一个像素进行检查，看这个点的梯度是不是周围具有相同梯度方向的点中最大的
    #    第四步滞后阈值
    #    要确定那些边界才是真正的边界
    #    这时我们需要设置两个阈值：minVal 和 maxVal
    #    当图像的灰度梯度高于 maxVal 时被认为是真的边界，那些低于 minVal 的边界会被抛弃
    #    如果介于两者之间的话，就要看这个点是否与某个被确定为真正的边界点相连
    #    如果是就认为它也是边界点，如果不是就抛弃

    #    第一个参数是输入图像
    #    第二和第三个分别是 minVal 和 maxVal(minVal,maxVal谁大谁小结果一样)
    #    第三个参数设置用来计算图像梯度的 Sobel
    #    卷积核的大小，默认值为 3
    #    最后一个参数是 L2gradient，用来设定求梯度大小的方程
    edge = cv2.Canny(img, minVal, maxVal)
    cv2.imshow('canny', edge)
cv2.destroyAllWindows()

image = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('origin', image)

h, w = image.shape   # 获取图像的高度和宽度


# Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(image, cv2.CV_16S, kernelx)
y = cv2.filter2D(image, cv2.CV_16S, kernely)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow('Roberts', Roberts )

# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(image, cv2.CV_16S, kernelx)
y = cv2.filter2D(image, cv2.CV_16S, kernely)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow('Prewitt', Prewitt)

# Sobel 滤波器 进行边的检测
sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # 水平方向
sobel_vetical = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # 垂直方向
cv2.imshow('sobel_H', sobel_horizontal)    # 水平方向
cv2.imshow('sobel_V', sobel_vetical)    # 垂直方向

# 拉普拉斯算子 进行边的检测    64F代表每一个像素点元素占64位浮点数
laplacian = cv2.Laplacian(image, cv2.CV_64F,  ksize=5)
cv2.imshow('laplacian', laplacian)

# # Canny边检测器
canny = cv2.Canny(image, 50, 240)
cv2.imshow('Canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()