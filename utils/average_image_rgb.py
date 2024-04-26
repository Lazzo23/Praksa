import cv2  # Import the OpenCV library for image processing.
import os   # Import the os module for interacting with the file system.
import numpy as np   # Import numpy for numerical operations.
import matplotlib.pyplot as plt   # Import matplotlib for data visualization.

# Directory where the images are stored.
dir_path = "images/"

# Initialize an empty list to store the images.
images = []

# Loop through each image in the directory.
for image_path in os.listdir(dir_path):
    # Read each image and convert its color format from BGR to RGB.
    image = cv2.cvtColor(cv2.imread(dir_path + image_path), cv2.COLOR_BGR2RGB)
    # Append the converted image to the list.
    images.append(image)

# Calculate the average image by taking the mean along the 0th axis (along images).
averageIm = np.mean(images, axis=0)

# Define a tuple of colors for plotting RGB channels.
barve = ('r', 'g', 'b')

# Create a figure for plotting histograms with specific size.
plt.figure(figsize=(20, 4))

# Loop through each RGB channel.
for i in range(3):
    # Plot a histogram for the i-th RGB channel of the average image.
    plt.hist(averageIm[:,:,i].ravel(), bins=256, range=(0, 255), color=barve[i], alpha=0.5, label=f'{barve[i]} channel')

# Add legend and axis labels to the plot.
plt.legend()  # Add legend based on label in the plot.
plt.title('Distribution of RGB pixels on average image')  # Set title for the plot.
plt.xlabel('Intensity')  # Set label for x-axis.
plt.ylabel('Number of pixels')  # Set label for y-axis.
plt.grid(True)  # Add grid to the plot.
plt.show()  # Display the plot.
