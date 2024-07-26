import numpy as np
import cv2

im = cv2.imread('/Users/vastavbharambe/Downloads/cap/test_images/2.png')
row, col = im.shape[:2]
bottom = im[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

border_size = 30
border = cv2.copyMakeBorder(
    im,
    top=border_size,
    bottom=border_size,
    left=border_size,
    right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

# cv2.imshow('image', im)
# cv2.imshow('bottom', bottom)
cv2.imshow('border', border)
cv2.imwrite('2ef.png',border )
cv2.waitKey(0)
cv2.destroyAllWindows()