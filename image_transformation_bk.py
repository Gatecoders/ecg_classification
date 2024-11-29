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


def graph_extraction(input_image_path='img0/', output_image_path='out_img/'):
    # Ensure the output directory exists
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    
    # Process each image in the input directory
    for filename in os.listdir(input_image_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('jpeg'):
            # Load the image
            image = cv2.imread(os.path.join(input_image_path, filename))
            
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
                output_filename = os.path.join(output_image_path, f'{filename}')
                cv2.imwrite(output_filename, graph_roi)

def resize_and_save_images(input_image_path='img/', output_image_path='out_img/', new_width=1300, new_height=1300):
    # Ensure output folder exists
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    # Loop through all files in the input directory
    for image_file in os.listdir(input_image_path):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            # Read the image
            image = imread(os.path.join(input_image_path, image_file))

            # Resize the image
            resized_image = resize(image, (new_height, new_width))

            # Ensure image is in uint8 format (required by imsave)
            resized_image = (resized_image * 255).astype(np.uint8)

            # Construct output image filename
            output_file = os.path.splitext(image_file)[0] + '_resized.jpg'
            output_file_path = os.path.join(output_image_path, output_file)

            # Save the resized image
            imsave(output_file_path, resized_image)


def Convert_Image_Lead(image_file, parent_folder):
    # Read the image
    image = imread(os.path.join(parent_folder, image_file), plugin='matplotlib')
    # Dividing the ECG leads from 1-13 from the above image
    Lead_1 = image[300:600, 150:643]
    Lead_2 = image[300:600, 646:1135]
    Lead_3 = image[300:600, 1140:1626]
    Lead_4 = image[300:600, 1630:2125]
    Lead_5 = image[600:900, 150:643]
    Lead_6 = image[600:900, 646:1135]
    Lead_7 = image[600:900, 1140:1626]
    Lead_8 = image[600:900, 1630:2125]
    Lead_9 = image[900:1200, 150:643]
    Lead_10 = image[900:1200, 646:1135]
    Lead_11 = image[900:1200, 1140:1626]
    Lead_12 = image[900:1200, 1630:2125]
    Lead_13 = image[1250:1480, 150:2125]

    # List of leads
    Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

    # Folder name to store lead images
    folder_name = re.sub('.jpg', '', image_file)

    # Ensure the directory exists before saving the lead images
    lead_dir = os.path.join(parent_folder, folder_name)
    if not os.path.exists(lead_dir):
        os.makedirs(lead_dir)

    # Loop through leads and create separate images
    for x, y in enumerate(Leads):
        fig, ax = plt.subplots()
        ax.imshow(y)
        ax.axis('off')
        ax.set_title("Leads {}".format(x + 1))
        plt.close('all')
        plt.ioff()

        # Save the image
        fig.savefig(os.path.join(lead_dir, 'Lead_{}_Signal.png'.format(x + 1)))

    extract_signal_leads(Leads, folder_name, parent_folder)


# Extract only signal from images
def extract_signal_leads(Leads, folder_name, parent):
    # Loop through image list containing all leads from 1-13
    for x, y in enumerate(Leads):
        # Creating subplot
        fig1, ax1 = plt.subplots()

        # Convert to grayscale
        grayscale = color.rgb2gray(y)

        # Smooth the image using Gaussian filter
        blurred_image = gaussian(grayscale, sigma=0.7)

        # Perform Otsu thresholding to create a binary image
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh

        # Resize image
        if x != 12:
            binary_global = resize(binary_global, (300, 450))

        ax1.imshow(binary_global, cmap="gray")
        ax1.axis('off')
        ax1.set_title("pre-processed Leads {} image".format(x + 1))
        plt.close('all')
        plt.ioff()

        # Ensure the directory exists before saving the image
        lead_dir = os.path.join(parent, folder_name)
        if not os.path.exists(lead_dir):
            os.makedirs(lead_dir)

        # Save the image
        fig1.savefig(os.path.join(lead_dir, 'Lead_{}_preprocessed_Signal.png'.format(x + 1)))

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
        ax7.set_title("Contour {} image".format(x + 1))
        plt.close('all')
        plt.ioff()

        # Save the contour image
        fig7.savefig(os.path.join(lead_dir, 'Lead_{}_Contour_Signal.png'.format(x + 1)))

        lead_no = x
        scale_csv_1D(test, lead_no, folder_name, parent)


def scale_csv_1D(test, lead_no, folder_name, parent):
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
    fig6.savefig(os.path.join(lead_dir, 'ID_Lead_{}_Signal.png'.format(lead_no + 1)))

    Normalized_Scaled = Normalized_Scaled.T
    Normalized_Scaled['Target'] = target
    # Save the scaled data to CSV
    csv_path = os.path.join(parent, 'scaled_data_1D_{}.csv'.format(lead_no + 1))
    if os.path.isfile(csv_path):
        Normalized_Scaled.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        Normalized_Scaled.to_csv(csv_path, index=False)


# Load the different types of folders
resize_and_save_images()
Path_to_img = 'out_img/'

# Types of heart. Now taking only 3. will work on COVID-19 later
Types_ECG = {'Path_to_img': Path_to_img}

# Loop through folder/files and create separate images of different leads
for types, folder in Types_ECG.items():
    for files in os.listdir(folder):
        if files.endswith(".jpg"):
            Convert_Image_Lead(files, folder)