""" Libraries"""

import os
import sys
import torch
import time as t
import numpy as np
import cv2
from skimage import transform
from torch.nn import functional as F
from segment_anything import sam_model_registry, SamPredictor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import (QBrush, QPainter, QPen, QPixmap, QKeySequence, QPen, QBrush, QColor, QImage, QIcon)
from PyQt5.QtWidgets import (QFileDialog, QApplication, QGraphicsScene, QGraphicsView, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QShortcut, QGridLayout, QComboBox, QLabel, QLineEdit)
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks



""" Glare Detection """

def green_glare(rgb_frame, smoothing_kernel=np.hanning(11)/5, ASF_sizes=(5, 3, 3, 1), dilations=(5, 3), min_threshold=215, coeff_tophat=3.25, tophat_kernel=6):
    '''
        Implementation by IBM, based on the main principle of [1] it takes a RGB image
        as input and uses the green channel to compute a binary mask of the big
        saturated and small bright regions that are supposed to correspond to glares.

        References
        ----------
        [1] Holger  Lange.   Automatic  glare  removal  in  reflectance  imagery
            of  the  uterine  cervix. Progress  in Biomedical Optics and Imaging
            - Proceedings of SPIE, 5747, 04 2005.
    ''' 
    MIN_FEATURE = 125

    feature = rgb_frame[:,:,1]
    histogram = np.bincount(np.ravel(feature), minlength=256)
    smoothed_histo = convolve1d(1.0 * histogram, smoothing_kernel, mode="reflect")
    a_min, _ = find_peaks(-smoothed_histo, distance=10, prominence=7)
    if a_min.size == 0: threshold = min_threshold
    else: threshold = max(min_threshold, a_min.max())
    saturated_mask = (feature >= threshold).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(feature, cv2.MORPH_TOPHAT, kernel)
    _, small_bright_mask = cv2.threshold(
        tophat * (feature >= MIN_FEATURE),
        coeff_tophat * np.mean(tophat ** 2),
        255,
        cv2.THRESH_BINARY,
    )
    small_bright_mask = small_bright_mask.astype("uint8")
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ASF_sizes[0], ASF_sizes[0]))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ASF_sizes[1], ASF_sizes[1]))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ASF_sizes[2], ASF_sizes[2]))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ASF_sizes[3], ASF_sizes[3]))
    filtered_saturated = cv2.morphologyEx(cv2.morphologyEx(saturated_mask, cv2.MORPH_CLOSE, kernel1), cv2.MORPH_OPEN, kernel2)
    filtered_small_bright = cv2.morphologyEx(cv2.morphologyEx(small_bright_mask, cv2.MORPH_CLOSE, kernel3), cv2.MORPH_OPEN, kernel4)
    enlarged_saturated_mask = cv2.dilate(filtered_saturated, np.ones((dilations[0], dilations[0]), np.uint8))
    enlarged_small_bright_mask = cv2.dilate(filtered_small_bright, np.ones((dilations[1], dilations[1]), np.uint8))
    mask = enlarged_saturated_mask | enlarged_small_bright_mask
    return mask



""" Colors for masks"""

colors = [
    (255, 0, 0), 
    (0, 255, 0), 
    (0, 0, 255), 
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]



class Window(QWidget):

    """ init """

    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowIcon(QIcon("icons/program_icon.png"))
        self.setWindowTitle("Segmentation Program")
        self.resize(800, 600)
        
        # Variables: Image
        self.IMAGE = None
        self.IMAGE_WIDTH = None
        self.IMAGE_HEIGHT = None
        self.SCENE = None
        self.IMAGE_PIXMAP = None

        # Variables: Masks
        self.MASK_CURRENT = None
        self.MASK_PREVIOUS = None
        self.PREVIOUS_MASKS = []

        # Variables: Segmentation Models
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MEDSAM_MODEL = None
        self.SAM_MODEL = None
        self.SAM_PREDICTOR = None
        self.IMAGE_EMBEDDING = None

        # Variables: Drawing and Options
        self.half_point_size = 5
        self.color_idx = 1
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.glare = True
        self.lines = None
        self.inverted = False
        self.dashed_line_item = None
        self.dashed_line_items = []

        # Grid Layout
        grid_layout = QGridLayout(self)

        # View
        self.view = QGraphicsView()
        self.view.setMouseTracking(True) 
        self.view.setRenderHint(QPainter.Antialiasing)

        # Adding view to layout
        grid_layout.addWidget(self.view, 1, 0) 

        # Labels
        label1 = QLabel("Select Segmentation Model: ")
        label2 = QLabel("Options: ")
        label3 = QLabel("Annotate as: ")
        
        # Buttons
        self.load_button = QPushButton("Load Image")
        self.medsam_button = QPushButton("MedSAM")
        self.medsam_button.setCheckable(True)
        self.sam_vitb_button = QPushButton("SAM ViT-b")
        self.sam_vitb_button.setCheckable(True)
        self.invert_button = QPushButton("Invert Mask")
        self.glare_button = QPushButton("Detect Glare")
        self.save_button = QPushButton("Save Mask")
        self.reset_button = QPushButton("Reset Image")

        # Dropdown menu for annotation types
        self.annotations_dropdown = QComboBox()
        self.annotations_dropdown.addItems(["Healthy", "Benign", "Cancerous", "Other"])
        self.annotation = self.annotations_dropdown.currentText()

        # Input text for "Other" annotation type
        self.input_text = QLineEdit(self)
        self.input_text.setMaximumWidth(194)
        self.input_text.setVisible(False)

        # Handlers for button events
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_mask)
        self.medsam_button.clicked.connect(self.select_medsam)
        self.sam_vitb_button.clicked.connect(self.select_sam)
        self.invert_button.clicked.connect(self.invert_mask)
        self.glare_button.clicked.connect(self.detect_glare)
        self.reset_button.clicked.connect(self.reset_image)
        self.annotations_dropdown.currentIndexChanged.connect(self.change_annotation)
        self.input_text.textChanged.connect(self.get_input_text)

        # Header
        vbox_buttons_top = QHBoxLayout()
        vbox_buttons_top.addWidget(self.load_button)
        vbox_buttons_top.addWidget(self.save_button)
        grid_layout.addLayout(vbox_buttons_top, 0, 0, 1, 1, alignment=Qt.AlignTop)

        # Sidebar 
        hbox_buttons_right = QVBoxLayout()
        hbox_buttons_right.addWidget(label1)
        hbox_medsam_sam_vitb = QHBoxLayout()
        hbox_medsam_sam_vitb.addWidget(self.medsam_button)
        hbox_medsam_sam_vitb.addWidget(self.sam_vitb_button)
        hbox_buttons_right.addLayout(hbox_medsam_sam_vitb)
        hbox_buttons_right.addWidget(label2)
        hbox_buttons_right.addWidget(self.invert_button)
        hbox_buttons_right.addWidget(self.glare_button)
        hbox_buttons_right.addWidget(self.reset_button)
        hbox_buttons_right.addWidget(label3)
        hbox_buttons_right.addWidget(self.annotations_dropdown)
        hbox_buttons_right.addWidget(self.input_text)
        grid_layout.addLayout(hbox_buttons_right, 1, 1, 1, 1, alignment=Qt.AlignTop)

        # Set Layout
        self.setLayout(grid_layout)

        # Set icons
        self.load_button.setIcon(QIcon("icons/load_icon.png"))
        self.save_button.setIcon(QIcon("icons/save_icon.png"))
        self.medsam_button.setIcon(QIcon("icons/medsam_icon.png"))
        self.sam_vitb_button.setIcon(QIcon("icons/sam_icon.png"))
        self.invert_button.setIcon(QIcon("icons/invert_icon.png"))
        self.glare_button.setIcon(QIcon("icons/glare_icon.png"))
        self.reset_button.setIcon(QIcon("icons/reset_icon.png"))

        # Keyboard Shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_mask)

        # Load segmentation models
        self.MEDSAM_MODEL, _ = self.load_model(model_path = "models/medsam.pth")
        self.SAM_MODEL, self.SAM_PREDICTOR = self.load_model(model_path = "models/sam_vit_b.pth")
        self.select_medsam() # Default model is MedSAM

        # Load image
        self.load_image()



    """ Load Segmentation Model """

    def load_model(self, model_path, model_type="vit_b"):

        print("Loading model ...")

        print(f"Model path: {model_path}\nModel type: {model_type}\nDevice: {self.DEVICE}")

        # Record start time for model loading
        start_time = t.time()

        # Load the segmentation model from the specified path
        model = sam_model_registry[model_type](checkpoint=None)
        model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        model.to(self.DEVICE)

        # Create a predictor object for the loaded model
        predictor = SamPredictor(model)

        # Calculate and print the time taken to load the model
        load_time = round(t.time() - start_time, 3)
        print(f"Model loaded in {load_time} seconds\n")

        return model, predictor
    


    """ Segmentation Model Switcher """

    def select_medsam(self):

        # Select MedSAM Segmentation Model
        self.curr_model = "MedSAM"

        # Check MedSAM Button and uncheck SAM ViT-b button
        self.medsam_button.setChecked(True)
        self.sam_vitb_button.setChecked(False)

    def select_sam(self):

        # Select SAM ViT-b Segmentation Model
        self.curr_model = "SAM" 

        # Check SAM ViT-b Button and uncheck MedSAM button
        self.sam_vitb_button.setChecked(True)
        self.medsam_button.setChecked(False)



    """ Load Image """

    def load_image(self):

        # Create new directory for input images
        dir_path = "./images/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Get image path
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image to Segment", dir_path, "Image Files (*.png *.jpg)")

        # Check if
        if not file_path:
            print("No image path specified, program closing ...")
            print("Bye!")
            exit()

        print(f"Selected image: {file_path}\n")

        # Read image, width and height
        self.IMAGE = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, _ = self.IMAGE.shape

        # Get image embeddings for both models
        self.image_embedding_medsam()
        self.image_embedding_sam()

        # Set scene
        self.reset_image()



    """ Image Embedding for SAM """

    def image_embedding_sam(self):

        print("Image embedding for SAM...")

        start_time = t.time()

        self.SAM_PREDICTOR.set_image(self.IMAGE)

        embedding_time = round(t.time() - start_time, 3)

        print(f"Image embedded in {embedding_time} seconds.\n")



    """ 
        Image Embedding for MedSAM 

        functions image_embedding_medsam() and medsam_inference() were mostly copied from 
        https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py, beacuse MedSAM model 
        predictions didn't work fine with default SAM ViT-b .set_image() and .predict() functions.

    """

    @torch.no_grad()
    def image_embedding_medsam(self):

        print("Image embedding for MedSAM...")

        start_time = t.time()

        img_1024 = transform.resize(self.IMAGE, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)  # normalize to [0, 1], (H, W, 3)

        # convert the shape to (3, H, W)
        img_1024_tensor = (torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.DEVICE))

        # if self.IMAGE_EMBEDDING is None:
        with torch.no_grad():
            self.IMAGE_EMBEDDING = self.MEDSAM_MODEL.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        
        embedding_time = round(t.time() - start_time, 3)

        print(f"Image embedded in {embedding_time} seconds.\n")

    @torch.no_grad()
    def medsam_inference(self, medsam_model, img_embed, box_1024, height, width):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(low_res_pred, size=(height, width), mode="bilinear", align_corners=False,)  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

        return medsam_seg



    """ Show Mask """

    def show_mask(self, append=True):
        
        # Add new mask to the stack unless show_mask() is called by undo() function
        if append:
            self.PREVIOUS_MASKS.append(self.MASK_CURRENT.copy())

        # Set blend parameter
        alpha = 0.8

        # Blend the original image with the updated mask
        blended_image = cv2.addWeighted(self.IMAGE, alpha, self.MASK_CURRENT, 1 - alpha, 0)

        # Remove the previous background image from the scene
        self.SCENE.removeItem(self.IMAGE_PIXMAP)

        # Add the blended image as the new background
        self.IMAGE_PIXMAP = self.SCENE.addPixmap(self.np2pixmap(np.array(blended_image)))



    """ Convert image to pixmap"""

    def np2pixmap(self, np_img):
        h, w, _ = np_img.shape
        bytesPerLine = 3 * w
        qImg = QImage(np_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg)



    """ Invert Mask """

    def invert_mask(self):
        
        # Find the indices of black pixels in the mask
        unsegmented_mask = np.all(self.MASK_CURRENT == [0, 0, 0], axis=-1)

        # Set colors for unsegmented areas and segmented areas
        self.MASK_CURRENT[unsegmented_mask] = colors[self.color_idx % len(colors)]
        self.MASK_CURRENT[~unsegmented_mask] = [0, 0, 0]

        # Show mask
        self.show_mask()



    """ Detect Glare on the Image"""

    def detect_glare(self):

        # Detect glare in the image
        glare_mask = green_glare(self.IMAGE)

        # Convert glare mask to RGB format
        mask_color = cv2.cvtColor(glare_mask, cv2.COLOR_GRAY2RGB)

        # Invert the colors of the mask
        mask_inv = cv2.bitwise_not(mask_color)

        # Check for black pixels in the inverted mask
        black_mask = np.all(mask_inv == [0, 0, 0], axis=-1)

        # Update the current mask with detected glare areas
        self.MASK_CURRENT[black_mask != 0] = colors[self.color_idx % len(colors)]

        # Show glare
        self.show_mask()



    """ Reset Scene (image and masks) """

    def reset_image(self):

        # Convert numpy array to QPixmap
        pixmap = self.np2pixmap(self.IMAGE)

        # Create a new QGraphicsScene with the image dimensions
        self.SCENE = QGraphicsScene(0, 0, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        
        # Reset variables for drawing
        self.end_point = None
        self.rect = None
        
        # Add background image to the scene
        self.IMAGE_PIXMAP = self.SCENE.addPixmap(pixmap)
        self.IMAGE_PIXMAP.setPos(0, 0)
        
        # Reset the color mask to zeros
        self.MASK_CURRENT = np.zeros((*self.IMAGE.shape[:2], 3), dtype="uint8")

        # Reset mask history
        self.PREVIOUS_MASKS = []
        
        # Set the new scene for the QGraphicsView
        self.view.setScene(self.SCENE)

        # Set event handlers for mouse events
        self.SCENE.mousePressEvent = self.mouse_press
        self.SCENE.mouseMoveEvent = self.mouse_move
        self.SCENE.mouseReleaseEvent = self.mouse_release



    """ Change Annotation Type From Dropdown Menu"""

    def change_annotation(self):

        # "Other" annotation type is selected
        if self.annotations_dropdown.currentText() == "Other":

            # Store annotation for file name in save_mask()
            self.annotation = self.input_text.text()

            # Input for text is visible
            self.input_text.setVisible(True)

        # Healthy, Benign or Cancerous anottation type is selected
        else:

            # Store annotation for file name in save_mask()
            self.annotation = self.annotations_dropdown.currentText()

            # Input for text is hidden
            self.input_text.setVisible(False)



    """ Real-Time Input Text For Annotation Type"""

    def get_input_text(self, input_text):
        self.annotation = input_text



    """ Save Mask """

    def save_mask(self):

        # Create new directory for output masks
        dir_path = "./output/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Allowed characters in annotation
        allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_.")

        # Check if annotation type is empty or uses unallowed characters
        if self.annotation == "" or not set(self.annotation).issubset(allowed_chars):
            
            # Reset input text
            self.input_text.setText("")
            print("Invalid Annotation Type.\n")
            return
        
        # Set image name
        image_path = f"{self.annotation}.png"

        # Check if image already exists
        if os.path.exists(os.path.join(dir_path, image_path)):
            
            # Change image name
            i = 1
            while True:
                i +=1 
                image_path = f"{self.annotation}{i}.png"
                if not os.path.exists(os.path.join(dir_path, image_path)):
                    break

        # Set output path                
        output_path = dir_path + image_path
            
        # Get color for the mask
        color = colors[self.color_idx % len(colors)]
        mask_color = np.array(color, dtype=np.uint8)
        
        # Fill holes in mask
        # filled_mask = cv2.morphologyEx(self.MASK_CURRENT, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        filled_mask = self.MASK_CURRENT.copy()
        
        # Create a mask with the specified color
        mask_color = np.tile(mask_color, (filled_mask.shape[0], filled_mask.shape[1], 1))
        mask = cv2.inRange(filled_mask, mask_color, mask_color)
        
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(self.IMAGE, self.IMAGE, mask=mask)
        
        # Save the masked image
        saved = cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

        # Check if writing was successfull
        if not saved:
            print("Error: Image not saved.")
            return
        
        print(f"Mask saved to: {output_path}\n")

        # Reset input text
        self.input_text.setText("")

        # Increment color index for the next mask
        # self.color_idx += 1

    

    """ Undo Mask """

    def undo(self):

        # Check if there are previous masks to undo
        if len(self.PREVIOUS_MASKS) >= 2:
            self.PREVIOUS_MASKS.pop()                                                   # Remove current mask
            self.MASK_CURRENT = self.PREVIOUS_MASKS[-1]                                 # Current mask = Previous mask
        
        elif len(self.PREVIOUS_MASKS) == 1:
            self.PREVIOUS_MASKS.pop()                                                   # Remove current mask
            self.MASK_CURRENT = np.zeros((*self.IMAGE.shape[:2], 3), dtype="uint8")     # Current mask = Default mask
        
        # No previous masks
        else:
            print("No previous mask record.\n")
            return
        
        # Show previous mask
        self.show_mask(append=False)



    """ Mouse Press Handler """

    def mouse_press(self, ev):
        
        # Get x, y coordinates for start point
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.start_pos = x, y

        # Reset mouse state
        self.is_mouse_down = True

        # Draw start point with any mouse button
        self.start_point = self.SCENE.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("black")),
            brush=QBrush(QColor("black")))



    """ Mouse Move Handler """

    def mouse_move(self, ev):
        # Get the x and y coordinates of the mouse pointer in the scene
        x, y = ev.scenePos().x(), ev.scenePos().y()

        # Remove the previous dashed lines, if any
        if self.dashed_line_items:
            for item in self.dashed_line_items:
                self.SCENE.removeItem(item)
            self.dashed_line_items.clear()

        # Check if the mouse pointer is within the bounds of the image
        if 0 <= x <= self.IMAGE_WIDTH and 0 <= y <= self.IMAGE_HEIGHT:
            # Determine the starting point of the line at the current cursor position
            start_point = ev.scenePos()

            # Determine the end points of the lines in different directions based on the size and position of the scene
            end_point_left = QPointF(0, y)
            end_point_right = QPointF(self.SCENE.width(), y)
            end_point_top = QPointF(x, 0)
            end_point_bottom = QPointF(x, self.SCENE.height())

            # Add dashed lines to the graphics scene
            self.dashed_line_items.append(self.SCENE.addLine(
                start_point.x(), start_point.y(),
                end_point_left.x(), end_point_left.y(),
                pen=QPen(Qt.black, 1, Qt.DashLine)
            ))
            self.dashed_line_items.append(self.SCENE.addLine(
                start_point.x(), start_point.y(),
                end_point_right.x(), end_point_right.y(),
                pen=QPen(Qt.black, 1, Qt.DashLine)
            ))
            self.dashed_line_items.append(self.SCENE.addLine(
                start_point.x(), start_point.y(),
                end_point_top.x(), end_point_top.y(),
                pen=QPen(Qt.black, 1, Qt.DashLine)
            ))
            self.dashed_line_items.append(self.SCENE.addLine(
                start_point.x(), start_point.y(),
                end_point_bottom.x(), end_point_bottom.y(),
                pen=QPen(Qt.black, 1, Qt.DashLine)
            ))

        # Check if the mouse button is not pressed
        if not self.is_mouse_down:
            return

        # Remove the previous endpoint if it exists
        if self.end_point is not None:
            self.SCENE.removeItem(self.end_point)

        # Add a new endpoint (ellipse) at the current mouse position
        self.end_point = self.SCENE.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("black")),
            brush=QBrush(QColor("black")),
        )

        # Remove the previous rectangle if it exists
        if self.rect is not None:
            self.SCENE.removeItem(self.rect)

        # Determine the minimum and maximum coordinates for the rectangle
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        # Add a rectangle to the scene
        self.rect = self.SCENE.addRect(xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("black")))

        

    """ Mouse Release """

    def mouse_release(self, ev):
        
        # Reset mouse state
        self.is_mouse_down = False

        # Get Bounding Box coordinates
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos

        # Ensure bounding box coordinates are within image boundaries
        xmin = max(min(x, sx), 0)
        xmax = min(max(x, sx), self.IMAGE_WIDTH)
        ymin = max(min(y, sy), 0)
        ymax = min(max(y, sy), self.IMAGE_HEIGHT)

        # Bounding box for prompt
        box_np = np.array([[xmin, ymin, xmax, ymax]])

        # Check if start and end rectangle points are the same
        if xmin == xmax and ymin == ymax:
            
            # Print error message
            print(f"Invalid Prompt: bounding box: {box_np}\n")
            
            # Remove start point from scene
            self.SCENE.removeItem(self.start_point)
            self.start_point = None
            
            return

        # Check if bounding box is outside of the scene
        if xmin > xmax or ymin > ymax:
            
            # Print error message
            print(f"Invalid Prompt: bounding box: {box_np}\n")

            # Remove start point, rectangle, end point
            self.SCENE.removeItem(self.start_point)
            self.start_point = None
            self.SCENE.removeItem(self.end_point)
            self.end_point = None
            self.SCENE.removeItem(self.rect)
            self.rect = None

            return

        print(f"Prompt: bounding box: {box_np}\n")

        # Predict mask based on current model
        if self.curr_model == "MedSAM":
            
            # Resize bounding box prompt for MedSAM
            box_1024 = box_np / np.array([self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_HEIGHT]) * 1024

            # Predict MedSAM mask
            mask = self.medsam_inference(self.MEDSAM_MODEL, self.IMAGE_EMBEDDING, box_1024, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        
        else:
            
            # Predict SAM ViT-b mask
            masks, _, _ = self.SAM_PREDICTOR.predict(point_coords = None, point_labels = None, box=box_np[None, :], multimask_output=False)
            mask = masks[0]

        # Update mask with current color index
        self.MASK_CURRENT[mask != 0] = colors[self.color_idx % len(colors)]

        # Display updated mask
        self.show_mask()

        # Remove graphical elements associated with bounding box
        self.SCENE.removeItem(self.start_point)
        self.start_point = None
        self.SCENE.removeItem(self.end_point)
        self.end_point = None
        self.SCENE.removeItem(self.rect)
        self.rect = None



""" Main """

app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()