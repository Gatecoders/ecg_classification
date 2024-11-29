import os
import re
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage import measure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import imread, imsave


class Transformation():
    def __init__(self, input_image_path='img0/', output_image_path='out_img/'):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.process_single_image()

    def process_single_image(self):
        # Ensure the output directory exists
        if not os.path.exists(self.output_image_path):
            os.makedirs(self.output_image_path)
        
        # Choose the image to process (modify this as needed)
        filename = 'your_image.jpg'  # Replace with the actual image file name

        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # Load the image
            image = cv2.imread(os.path.join(self.input_image_path, filename))
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding to binarize the image
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Proceed only if there are contours found
            if len(contours) > 0:
                # Sort contours by area and get the largest one
                largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                
                # Get the bounding box of the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extract the region of interest (ROI) corresponding to the graph
                graph_roi = image[y:y+h, x:x+w]
                
                # Save the extracted graph
                output_filename = os.path.join(self.output_image_path, f'{filename}')
                cv2.imwrite(output_filename, graph_roi)

                # Resize the saved image
                self.resize_and_save_image(output_filename)

                # Convert image to leads
                self.convert_image_to_leads(output_filename)

    def resize_and_save_image(self, image_path, new_width=2213, new_height=1572):
        # Read the image
        image = imread(image_path)

        # Resize the image
        resized_image = resize(image, (new_height, new_width))

        # Ensure image is in uint8 format (required by imsave)
        resized_image = (resized_image * 255).astype(np.uint8)

        # Construct output image filename
        output_file = os.path.splitext(os.path.basename(image_path))[0] + '_resized.jpg'
        output_file_path = os.path.join(self.output_image_path, output_file)

        # Save the resized image
        imsave(output_file_path, resized_image)

    def convert_image_to_leads(self, image_path):
        # Read the image
        image = imread(image_path)

        # Dividing the ECG leads from 1-13 from the above image
        Lead_1 = image[50:360, 10:570]
        Lead_2 = image[50:360, 580:1110]
        Lead_3 = image[50:360, 1125:1640]
        Lead_4 = image[50:360, 1665:2240]
        Lead_5 = image[380:750, 10:570]
        Lead_6 = image[380:750, 580:1110]
        Lead_7 = image[380:750, 1125:1640]
        Lead_8 = image[380:750, 1665:2240]
        Lead_9 = image[760:1100, 10:570]
        Lead_10 = image[760:1100, 580:1110]
        Lead_11 = image[760:1100, 1125:1640]
        Lead_12 = image[760:1100, 1665:2240]
        Lead_13 = image[1250:1580, 10:2240]

        # List of leads
        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

        # Folder name to store lead images
        folder_name = os.path.splitext(os.path.basename(image_path))[0]

        # Ensure the directory exists before saving the lead images
        lead_dir = os.path.join(self.output_image_path, folder_name)
        if not os.path.exists(lead_dir):
            os.makedirs(lead_dir)

        # Loop through leads and create separate images
        for idx, lead_image in enumerate(Leads):
            fig, ax = plt.subplots()
            ax.imshow(lead_image)
            ax.axis('off')
            ax.set_title(f"Leads {idx + 1}")
            plt.close('all')
            plt.ioff()

            # Save the image
            fig.savefig(os.path.join(lead_dir, f'Lead_{idx + 1}_Signal.png'))

            # Call extract_signal_leads for further processing
            self.extract_signal_leads([lead_image], folder_name, self.output_image_path)

    def extract_signal_leads(self, leads, folder_name, parent):
        # Loop through image list containing all leads from 1-13
        for idx, lead_image in enumerate(leads):
            # Creating subplot
            fig1, ax1 = plt.subplots()

            # Convert to gray scale
            grayscale = color.rgb2gray(lead_image)

            # Smoothing image
            blurred_image = gaussian(grayscale, sigma=0.7)

            # Thresholding to distinguish foreground and background
            # Using Otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            # Creating binary image based on threshold
            binary_global = blurred_image < global_thresh

            # Resize image
            if idx != 12:
                binary_global = resize(binary_global, (300, 450))

            ax1.imshow(binary_global, cmap="gray")
            ax1.axis('off')
            ax1.set_title(f"pre-processed Leads {idx + 1} image")
            plt.close('all')
            plt.ioff()

            # Ensure the directory exists before saving the image
            lead_dir = os.path.join(parent, folder_name)
            if not os.path.exists(lead_dir):
                os.makedirs(lead_dir)

            # Save the image
            fig1.savefig(os.path.join(lead_dir, f'Lead_{idx + 1}_preprocessed_Signal.png'))

            fig7, ax7 = plt.subplots()
            plt.gca().invert_yaxis()

            # Find contour and get only the necessary signal contour
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([contour.shape for contour in contours])[::-1][0:1]
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
                    ax7.plot(test[:, 1], test[:, 0], linewidth=1, color='black')
            ax7.axis('image')
            ax7.set_title(f"Contour {idx + 1} image")
            plt.close('all')
            plt.ioff()

            # Save the contour image
            fig7.savefig(os.path.join(lead_dir, f'Lead_{idx + 1}_Contour_Signal.png'))

            lead_no = idx
            self.scale_csv_1D(test, lead_no, folder_name, parent)

    def scale_csv_1D(self, test, lead_no, folder_name, parent):
        target = folder_name[0:2]
        # Scaling the data
        scaler = MinMaxScaler()
        fit_transform_data = scaler.fit_transform(test)
        Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])
        fig6, ax6 = plt.subplots()
        plt.gca().invert_yaxis()
        ax6.plot(Normalized_Scaled, linewidth=1, color='black', linestyle='solid')
        plt.close('all')
        plt.ioff()

        # Ensure the directory exists before saving the scaled signal image
        lead_dir = os.path.join(parent, folder_name)
        if not os.path.exists(lead_dir):
            os.makedirs(lead_dir)

        # Save the scaled signal image
        fig6.savefig(os.path.join(lead_dir, f'ID_Lead_{lead_no + 1}_Signal.png'))

        Normalized_Scaled = Normalized_Scaled.T
        Normalized_Scaled['Target'] = target
        # Save the scaled data to CSV
        csv_path = os.path.join(parent, f'scaled_data_1D_{lead_no + 1}.csv')
        if os.path.isfile(csv_path):
            Normalized_Scaled.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            Normalized_Scaled.to_csv(csv_path, index=False)
