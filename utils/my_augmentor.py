import cv2
import os
import numpy as np

class MyAugmentor:

    def __init__(self, images_path, masks_path):
        self.i = 0
        self.images = self.get_files(images_path)
        self.masks = self.get_files(masks_path)
        self.generated_images = []
        self.generated_masks = []

    """ Read images from directory"""
    def get_files(self, dir_path):
        files = []

        for file_path in os.listdir(dir_path):
            file = cv2.imread(dir_path + file_path)
            files.append(file)

        return files
    
    """ Save generated images and masks """
    def write_files(self, images_path, masks_path):
        for i, (image, mask) in enumerate(zip(self.generated_images, self.generated_masks)):
            cv2.imwrite(f'{images_path}image_{i}.png', image)
            cv2.imwrite(f'{masks_path}mask_{i}.png', mask)
        
        print("Writing Successful!")
    
    """ Generate new images and masks from originals"""
    def generate_files(self):

        for image, mask in zip(self.images, self.masks):
            
            # Add original image and mask to the list
            self.generated_images.extend([image])
            self.generated_masks.extend([mask])
            
            # Apply different filters to original image and add new image and its mask to the list
            filtered_images, filtered_masks = self.apply_filters(image, mask)
            self.generated_images.extend(filtered_images)
            self.generated_masks.extend(filtered_masks)
            
            # Horizontal flip image and mask and add to the list
            image, mask = self.flip(image, mask, 0)
            self.generated_images.extend([image])
            self.generated_masks.extend([mask])

            # Repeat filtering on flipped image and mask
            filtered_images, filtered_masks = self.apply_filters(image, mask)
            self.generated_images.extend(filtered_images)
            self.generated_masks.extend(filtered_masks)

            # Flip vertically
            image, mask = self.flip(image, mask, 1)
            self.generated_images.extend([image])
            self.generated_masks.extend([mask])

            # Repeat filtering
            filtered_images, filtered_masks = self.apply_filters(image, mask)
            self.generated_images.extend(filtered_images)
            self.generated_masks.extend(filtered_masks)

            # Flip horizontally again
            image, mask = self.flip(image, mask, 0)
            self.generated_images.extend([image])
            self.generated_masks.extend([mask])

            # Repeat filtering
            filtered_images, filtered_masks = self.apply_filters(image, mask)
            self.generated_images.extend(filtered_images)
            self.generated_masks.extend(filtered_masks)
    
    """ Flip image horizontaly or vertically"""
    def flip(self, image, mask, ax):
        image = np.flip(np.array(image), axis = ax)
        mask = np.flip(np.array(mask), axis = ax)

        return image, mask
    
    """ Applying different filters to the image and mask"""
    def apply_filters(self, image, mask):
        filtered_images = []
        filtered_masks = []

        # Sharpening
        filtered_images.append(self.sharpen(image))
        filtered_masks.append(mask)

        # Bluring
        filtered_images.append(self.blur(image))
        filtered_masks.append(mask)

        return filtered_images, filtered_masks

    def sharpen(self, image):
        filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = np.array(cv2.filter2D(image, -1, filter))

        return sharpened
    
    def blur(self, image):
        filter_size = (5, 5) 
        blurred = np.array(cv2.blur(image, filter_size))

        return blurred

aug = MyAugmentor("dir_1/", "dir_0/")
print(aug.masks)
aug.generate_files()
aug.write_files("dir_1/", "dir_0/")