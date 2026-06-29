from datasets import load_dataset
import cv2
import numpy as np

# Food101 데이터셋 불러오기
dataset = load_dataset("ethz/food101", split="train[:1]")

# 첫 번째 이미지 가져오기
image = np.array(dataset[0]["image"])

# RGB → BGR(OpenCV)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 1. 크기 조정
resize = cv2.resize(image, (224, 224))

# 2. Grayscale
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

# 3. Normalize
normalize = gray.astype(np.float32) / 255.0

# 4. Blur
blur = cv2.GaussianBlur(resize, (5, 5), 0)

# 5-1. 좌우 반전
flip = cv2.flip(resize, 1)

# 5-2. 회전 (30도)
height, width = resize.shape[:2]
matrix = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
rotate = cv2.warpAffine(resize, matrix, (width, height))

# 5-3. 색상 변화(밝기 증가)
bright = cv2.convertScaleAbs(resize, alpha=1.0, beta=50)

# 결과 출력
cv2.imshow("Original", resize)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Flip", flip)
cv2.imshow("Rotate", rotate)
cv2.imshow("Bright", bright)

cv2.waitKey(0)
cv2.destroyAllWindows()
