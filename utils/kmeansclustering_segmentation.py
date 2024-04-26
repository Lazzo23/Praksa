# Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
K-Means Clustering algoritem razdeli piksle v k gruč. Piksel je dodeljen tisti gruči, katere povprečje 
je najbližje vrednosti piksla. Algoritem se izvajaja iterirajoče (povprečja gruč se z dodeljevanjem pikslov 
popravljajo) dokler povprečje gruče ne skonvergira.
"""

# Read image and convert to RGB color space
path = './images/testna3.png'
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

"""
Algoritem potrebuje vektorizirano obliko slike, kjer so vsi piksli predstavljeni kot vrstie v matriki. 
To dosežemo z ukazom img.reshape((-1,3). V nadaljevanju bomo uporabili knjižnico cv2, ki pa potrebuje 
vrednosti v float obliki, zato začetne unit8 vrednosti pikslov spremenimo z ukazom np.float32().
"""

# Reshaping the image into a 2D array of pixels and 3 color channels
print(f'Original image: {img.shape}')
vectorized = np.float32(img.reshape((-1,3)))
print(f'Reshaped image: {vectorized.shape}')

"""
V spodnjem koraku definirmao podrobnosti izvajanja algortima. S k=4 nastavimo število gruč oziroma barv. 
S criteria nastavimo zaustavitvene pogoje, in sicer cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER 
zaustavi iteriranje, če je vsaj eden izmed dveh zgornji pogojev izpolnjen (algoritem doseže določeno 
natančnost - epsilon ali algoritem izvede vse iteracije).
"""

# Algorithm parameters
k = 3
num_iterations = 20
epsilon = 0.1
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iterations, epsilon)

"""
Z ukazom cv2.kmeans() poženemo algoritem. Kot argumente mu dodamo:
    - vectorized (slika)
    - k (število gruč)
    - None (začetne pozicije gruč)
    - 10 (število poskusov izbire najboljše začetne pozicije gruč)
    - cv2.KMEANS_RANDOM_CENTER (naključna inicializacija začetnih pozicij)
Algoritem vrne labels, ki za vsak piksel hrani njegovo gručo, in centers, ki hrani razrede oziroma barve.
"""

# Run the clustering
_, labels, centers = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

"""
Z ukazom np.uint8(centers) spremenimo vrednosti pikslov v cela števila in s pomočjo oznak oznak uveljavimo 
segmente na originalni sliki.
"""

# Apply segmentation to the image
centers = np.uint8(centers)
segmented = centers[labels.flatten()].reshape(img.shape)

# Show segmented image
plt.subplot(121)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img)
plt.subplot(122)
plt.axis('off')
plt.title(f'Segmented Image k = {k}')
plt.imshow(segmented)
plt.tight_layout()
plt.show()

segmented_images = []
for k in [3, 4, 5]:

    # Algorithm parameters
    num_iterations = 20
    epsilon = 0.1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iterations, epsilon)

    """
    Z ukazom cv2.kmeans() poženemo algoritem. Kot argumente mu dodamo:
        - vectorized (slika)
        - k (število gruč)
        - None (začetne pozicije gruč)
        - 10 (število poskusov izbire najboljše začetne pozicije gruč)
        - cv2.KMEANS_RANDOM_CENTER (naključna inicializacija začetnih pozicij)
    Algoritem vrne labels, ki za vsak piksel hrani njegovo gručo, in centers, ki hrani razrede oziroma barve.
    """

    # Run the clustering
    _, labels, centers = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    """
    Z ukazom np.uint8(centers) spremenimo vrednosti pikslov v cela števila in s pomočjo oznak oznak uveljavimo 
    segmente na originalni sliki.
    """

    # Apply segmentation to the image
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    segmented_images.append(segmented)

# Show segmented image
plt.subplot(141)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img)
plt.subplot(142)
plt.axis('off')
plt.title(f'Segmented Image k = {3}')
plt.imshow(segmented_images[0])
plt.subplot(143)
plt.axis('off')
plt.title(f'Segmented Image k = {4}')
plt.imshow(segmented_images[1])
plt.subplot(144)
plt.axis('off')
plt.title(f'Segmented Image k = {5}')
plt.imshow(segmented_images[2])
plt.tight_layout()
plt.show()