import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

""" Gaussian Noise """

# Load the image""
path = 'image.png'
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Function to add Gaussian noise to the image
def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image

# Add Gaussian noise to the image
noisy_image_gaussian = add_gaussian_noise(image, mean=0, std_dev=30)

# Display the original and corrupted images
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(noisy_image_gaussian)
axs[1].set_title('Gaussian Noise')
axs[1].axis('off')
plt.tight_layout()
plt.show()

""" Salt and Pepper Noise """

# Load the image
path = 'image.png'
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Function to add salt and pepper noise to the image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    height, width, channels = noisy_image.shape
    for i in range(height):
        for j in range(width):
            rand = random.random()
            if rand < salt_prob:
                noisy_image[i, j] = [255, 255, 255]  # Add salt noise
            elif rand > 1 - pepper_prob:
                noisy_image[i, j] = [0, 0, 0]  # Add pepper noise
    return noisy_image

# Add salt and pepper noise to the image
noisy_image_salt_pepper = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)


# Display the original and corrupted images
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(noisy_image_salt_pepper)
axs[1].set_title('Salt and Pepper Noise')
axs[1].axis('off')
plt.tight_layout()
plt.show()

""" Speckle Noise """

# Load the image
path = 'image.png'
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Add speckle noise
speckle = np.random.randn(*image.shape) * 0.1
noisy_image_speckle = np.clip(image + image * speckle, 0, 255).astype(np.uint8)

# Display the original and corrupted images
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(noisy_image_speckle)
axs[1].set_title('Speckle Noise')
axs[1].axis('off')
plt.tight_layout()
plt.show()

""" Median Blur """

# Display the original and corrupted images
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.medianBlur(noisy_image_gaussian, 5))
axs[1].set_title('medianBlur(Gaussian Noise)')
axs[1].axis('off')
axs[2].imshow(cv2.medianBlur(noisy_image_salt_pepper, 5))
axs[2].set_title('medianBlur(Salt and Pepper Noise)')
axs[2].axis('off')
axs[3].imshow(cv2.medianBlur(noisy_image_speckle, 5))
axs[3].set_title('medianBlur(Speckle Noise)')
axs[3].axis('off')
plt.tight_layout()
plt.show()

""" Gaussian Blur """

# Display the original and corrupted images
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.GaussianBlur(noisy_image_gaussian, (5, 5), 0))
axs[1].set_title('gaussianBlur(Gaussian Noise)')
axs[1].axis('off')
axs[2].imshow(cv2.GaussianBlur(noisy_image_salt_pepper, (5, 5), 0))
axs[2].set_title('gaussianBlur(Salt and Pepper Noise)')
axs[2].axis('off')
axs[3].imshow(cv2.GaussianBlur(noisy_image_speckle, (5, 5), 0))
axs[3].set_title('gaussianBlur(Speckle Noise)')
axs[3].axis('off')
plt.tight_layout()
plt.show()

""" FastNlMeansDenoisingColored Blur"""

# Display the original and corrupted images
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.fastNlMeansDenoising(noisy_image_gaussian, None, 10, 10, 7))
axs[1].set_title('fNDC(Gaussian Noise)')
axs[1].axis('off')
axs[2].imshow(cv2.fastNlMeansDenoising(noisy_image_salt_pepper, None, 10, 10, 7))
axs[2].set_title('fNDC(Salt and Pepper Noise)')
axs[2].axis('off')
axs[3].imshow(cv2.fastNlMeansDenoising(noisy_image_speckle, None, 10, 10, 7))
axs[3].set_title('fNDC(Speckle Noise)')
axs[3].axis('off')
plt.tight_layout()
plt.show()