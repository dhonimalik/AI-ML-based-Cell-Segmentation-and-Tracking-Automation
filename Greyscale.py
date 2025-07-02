import cv2
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

image = cv2.imread("img_8.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((4, 4), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)


sure_bg = cv2.dilate(opening, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)


_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = watershed(-dist_transform, markers, mask=opening)

segmented = np.zeros_like(gray)
for label in np.unique(markers):
    if label <= 1:
        continue
    segmented[markers == label] = 255

cv2.imwrite("cells_white_background_black.png", segmented)

plt.imshow(segmented, cmap='gray')
plt.title("Cells as White, Background as Black")
plt.axis('off')
plt.show()
