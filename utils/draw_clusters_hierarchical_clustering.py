import pandas as pd
import numpy as np
import cv2
import random

"""
Function "draw_clusters(clusters_path, result_path)"
reads pixel's cluster associated by hierarchical 
clustering in Orange pipeline from .csv file. It creates
segmented image by drawing each pixel with cluster colour
and saves the result to "result_path".
"""

def draw_clusters(clusters_path, result_path):

    # Read .csv file with clustered pixels 
    clustered_pixels = pd.read_csv(clusters_path, sep=',')

    # Resized image width and height (for reconstructing back the segmented image!)
    width = 96
    height = 70

    # Prepare blank image for drawing pixels
    clustered_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Prepare colormap dict for clusters
    cluster_colours = {}

    # Calculating new random colour for new cluster
    def get_new_colour(colours):

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        new_colour = [r, g, b]

        # If colour does not exists yet, return it, else create new one
        if new_colour not in colours:
            return new_colour
        
        get_new_colour(colours)

    # Iterate over each pixel (skip first 3 rows which represent the header)
    for _, row in clustered_pixels.iloc[2:].iterrows():
        
        # Read pixel's position and its cluster
        x = int(row["x_position"])
        y = int(row["y_position"])
        cluster = row["Cluster"]

        # If there is a new cluster, get it new colour
        if cluster not in cluster_colours:
            cluster_colours[cluster] = get_new_colour(cluster_colours.values())

        # Colour the pixel with cluster's colour
        clustered_image[y, x] = cluster_colours[cluster]

    # Resize clustered image
    clustered_image_resized = cv2.resize(clustered_image, (480, 350), interpolation= cv2.INTER_LINEAR)

    # Save segmented image
    cv2.imwrite(result_path, clustered_image_resized)

    return clustered_image_resized

segmented_image = draw_clusters("files/clustered_pixels.csv", "output/clustered.png")