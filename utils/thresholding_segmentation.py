# Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.filters import threshold_otsu
import cv2

# Read image
img_bgr = cv2.imread('./images/testna.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 

# Normalize grayscale pixel values to [0, 1]
img_gray_norm = cv2.normalize(img_gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Testing different threshoulds for best binary segmentation
thresholds = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])

# Show segmented images
fig, ax = plt.subplots(2, 5, figsize=(15,8))
for th, ax in zip(thresholds, ax.flatten()):
    img_segmented = img_gray_norm < th
    ax.imshow(img_segmented)
    ax.set_title('$Threshold = %.2f$' % th)
    ax.axis('off')
fig.tight_layout()
fig.show()

# Calculate optimal threshold with Otsu method
threshold = threshold_otsu(img_gray_norm)
img_otsu  = img_gray_norm < threshold

# Prepare histogram
freq, bins = histogram(img_gray_norm)

# Filter RGB image
r = img_rgb[:,:,0] * img_otsu
g = img_rgb[:,:,1] * img_otsu
b = img_rgb[:,:,2] * img_otsu
filtered = np.dstack([r,g,b])

# Show segmented image
fig, ax = plt.subplots(1, 4, figsize=(20,6))

ax[0].set_title('Original Image')
ax[0].imshow(img_rgb)
ax[0].axis('off')
ax[1].set_title(f'$Otsu Threshold = {threshold}$')
ax[1].imshow(img_otsu)
ax[1].axis('off')
ax[2].set_title('Mask Overlay')
ax[2].imshow(filtered)
ax[2].axis('off')
ax[3].set_title('Pixel Intesity Distribution')
ax[3].step(bins, freq)
ax[3].axvline(x=threshold, color='r', linestyle='-')
ax[3].set_xlabel('Pixel Intesity')
ax[3].set_ylabel('Number of Pixels')
plt.show()

# Show segmented images
fig, ax = plt.subplots(1, 4, figsize=(15,4))
for i, a in enumerate(ax):

    # Read images and convert to correct shapes
    img_bgr = cv2.imread(f'./images/test{i+1}.png')
    img_bgr = cv2.resize(img_bgr, (480, 350))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray_norm = cv2.normalize(img_gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Find the optimal threshold and create Otsu mask
    threshold = threshold_otsu(img_gray_norm)
    img_otsu  = img_gray_norm < threshold

    # Combine Otsu mask with the image
    r = img_rgb[:,:,0] * img_otsu
    g = img_rgb[:,:,1] * img_otsu
    b = img_rgb[:,:,2] * img_otsu
    filtered = np.dstack([r, g, b])

    # Show result
    a.imshow(filtered)
    a.set_title(f'$Otsu Threshold = {threshold:.3f}$')
    a.axis('off')
plt.tight_layout()
plt.show()