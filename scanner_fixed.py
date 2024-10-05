import cv2
import imutils

org_img = cv2.imread('images\\1188-receipt.jpg')

# scaling larger
ratio = org_img.shape[0] / 500.0
resized_img = imutils.resize(org_img, height=500)

# make it gray
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
# cv2.imshow('Gray Image', gray_img)

# make it blured
blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
# cv2.imshow('Blurred image', blurred_img)

# make edges
canny_img = cv2.Canny(blurred_img, 55, 125, apertureSize=3, L2gradient=True)
cv2.imshow('Canny', canny_img)

# find contours
contours = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# approximate contours by line
approx_cont = []
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_cont.append(approx)

# approx_cont = sorted(approx_cont, key=cv2.contourArea, reverse=True)[:7]

cv2.drawContours(resized_img, approx_cont, -1, (0, 255, 0), 2)
cv2.imshow('Contours', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
