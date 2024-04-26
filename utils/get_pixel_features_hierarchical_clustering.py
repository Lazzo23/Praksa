import cv2
import numpy as np

'''

Function get_pixel_features() accepts image, image's ROIs data and 
path to output file. Output file is a .csv file prepared for 
hierarchical clustering in Orange pipeline. Each row contains 
features for one specific image pixel.

Features:
    x - coordinate x, 
    y - coordinate y, 
    r - value of RED channel, 
    g - value of GREEN channel, 
    b - value of BLUE channel, 
    intensity - intensity of grayscale image, 
    r_avg - average RED values of pixel's neighbours, 
    g_avg - average GREEN values of pixel's neighbours, 
    b_avg - average BLUE values of pixel's neighbours, 
    intensity_avg - average intesity of pixel's neighbours,

Pixel clustering will be done by comparing similarities based on 
pixel's features.

'''

def get_pixel_features(image_path, output_path):

    # Read image (BGR format)
    image = cv2.imread(image_path)

    # Resize image
    resize_factor = 5
    h, w, _ = image.shape 

    image = cv2.resize(image, (w // resize_factor, h // resize_factor), interpolation= cv2.INTER_LINEAR)

    # Another way of resizing the image using max-pool resizing
    # image = max_pool_resizing(image, (4, 4))

    # Convert image to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Calculate average color for each pixel
    image_avg = cv2.blur(image, (3, 3))

    # Calculate average intensity for each pixel
    image_gray_avg = cv2.blur(image_gray, (3, 3))

    # Prepare data array
    pixels_data = []

    # Open new .csv file
    with open(output_path, "w") as f:

        # Prepare header for .csv file ("C" stands for continuous-typed feature)
        header = ["C#x_position", 
                  "C#y_position", 
                  "C#r_value", 
                  "C#g_value", 
                  "C#b_value",
                  "C#intensity", 
                  "C#r_value_avg", 
                  "C#g_value_avg", 
                  "C#b_value_avg", 
                  "C#intensity_avg"]
        
        f.write(",".join(map(str, header)) + "\n")

        # Loop thorugh each pixel on the image
        height, width, _ = image.shape
        for row in range(height):
            for column in range(width):
                
                # Get specific pixel
                pixel = image[row, column]

                # Skip glare pixels
                if not np.any(pixel):
                    continue
                
                # Get pixel's XY positions
                x = column
                y = row

                # Get pixel's RGB values
                b = pixel[0]
                g = pixel[1]
                r = pixel[2]

                # Get intensity
                intensity = image_gray[row, column]

                # Get average RGB values of pixel's neighbours
                pixel_avg = image_avg[row, column]
                b_avg = pixel_avg[0]
                g_avg = pixel_avg[1]
                r_avg = pixel_avg[2]

                # Get average intensity of pixel's neighbours
                intensity_avg = image_gray_avg[row, column]

                # Append pixel's features to the list
                pixel_data = [x, 
                              y, 
                              r, 
                              g, 
                              b, 
                              intensity, 
                              r_avg, 
                              g_avg, 
                              b_avg, 
                              intensity_avg]
                
                pixels_data.append(pixel_data)

                f.write(",".join(map(str, pixel_data)) + "\n")

    return pixels_data

image_path = "images/image.png"
output_path = "output/pixels.csv"

pixels = get_pixel_features(image_path, output_path)
print(f"Number of pixels: {len(pixels)}")