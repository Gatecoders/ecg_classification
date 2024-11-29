import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage import color
from skimage import measure
from natsort import natsorted
from skimage.transform import resize
from skimage.io import imread, imsave
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu, gaussian
from config import train_image_dir,test_image_dir, train_csv_dir, test_csv_dir, pca_mdls


def graph_extraction(transform, model_name, model_type):
    pca_dir = pca_mdls
    input_image_path = ''
    output_file_path = ''
    if transform==0 or transform==2:
        input_image_path = train_image_dir
        output_file_path = train_csv_dir
    elif transform==1:
        input_image_path = test_image_dir
        output_file_path = test_csv_dir
    
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
                graph_roi = cv2.cvtColor(graph_roi, cv2.COLOR_BGR2RGB)

                resize_and_save_images(filename,graph_roi,output_file_path)
           
    return final_csv(output_file_path, transform, pca_dir, model_name, model_type)

def resize_and_save_images(filename,graph, output_file_path, new_width=2213, new_height=1572):
    # Ensure output folder exists
    
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Read the image
    image = graph

    # Resize the image
    resized_image = resize(image, (new_height, new_width))

    # Ensure image is in uint8 format (required by imsave)
    resized_image = (resized_image * 255).astype(np.uint8)

    Convert_Image_Lead(resized_image,filename,output_file_path)

def Convert_Image_Lead(image_file,filename, output_file_path):
    
    image = image_file

    # Dividing the ECG leads from 1-13 from the above image
    # for kaggle or github dataset
    Lead_1 = image[50:440, 80:600]
    Lead_2 = image[50:440, 610:1120]
    Lead_3 = image[50:440, 1122:1635]
    Lead_4 = image[50:440, 1642:2160]
    Lead_5 = image[440:760, 80:600]
    Lead_6 = image[440:760, 610:1120]
    Lead_7 = image[440:760, 1122:1635]
    Lead_8 = image[440:760, 1642:2160]
    Lead_9 = image[880:1250, 80:600]
    Lead_10 = image[880:1250, 610:1120]
    Lead_11 = image[880:1250, 1122:1635]
    Lead_12 = image[880:1250, 1642:2160]
    Lead_13 = image[1250:1580, 10:2240]

    # for client dataset
    # Lead_1 = image[50:440, 80:550]
    # Lead_2 = image[50:440, 610:1100]
    # Lead_3 = image[50:440, 1130:1635]
    # Lead_4 = image[50:440, 1700:2200]
    # Lead_5 = image[410:760, 80:550]
    # Lead_6 = image[410:760, 610:1100]
    # Lead_7 = image[410:760, 1130:1635]
    # Lead_8 = image[410:760, 1700:2200]
    # Lead_9 = image[700:1250, 80:550]
    # Lead_10 = image[700:1250, 610:1100]
    # Lead_11 = image[700:1250, 1130:1635]
    # Lead_12 = image[700:1250, 1700:2200]
    # Lead_13 = image[1250:1580, 10:2240]

    Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

    extract_signal_leads(Leads, filename, output_file_path)

# Extract only signal from images
def extract_signal_leads(Leads, filename, output_file_path):
    
    for x, y in enumerate(Leads):                       # Loop through image list containing all leads from 1-13
        grayscale = color.rgb2gray(y)
        
        blurred_image = gaussian(grayscale, sigma=0.7)  # Smoothing image
    
        global_thresh = threshold_otsu(blurred_image)   # Thresholding to distinguish foreground and background and Using Otsu thresholding for getting threshold value

        binary_global = blurred_image < global_thresh   # Creating binary image based on threshold

        if x != 12:                                      # Resize image
            binary_global = resize(binary_global, (300, 450))

        contours = measure.find_contours(binary_global, 0.8)
        contours_shape = sorted([contour.shape for contour in contours])[::-1][0:1]
        for contour in contours:
            if contour.shape in contours_shape:
                test = resize(contour, (255, 2))

        lead_no = x
        scale_csv_1D(test, lead_no, filename, output_file_path)


def scale_csv_1D(test, lead_no, filename, output_file_path):
    
    target = filename[0:2]
    
    scaler = MinMaxScaler()  # Scaling the data
    fit_transform_data = scaler.fit_transform(test)
    Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])

    Normalized_Scaled = Normalized_Scaled.T
    Normalized_Scaled['Target'] = target

    # Save the scaled data to CSV
    csv_path = os.path.join(output_file_path, 'scaled_data_1D_{}.csv'.format(lead_no + 1))
    if os.path.isfile(csv_path):
        Normalized_Scaled.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        Normalized_Scaled.to_csv(csv_path, index=False)

def final_csv(dir, transform, pca_dir, model_name, model_type):
    
    location= dir
    test_final=pd.DataFrame()
    target=pd.DataFrame()
    output_dir = dir

    for files in natsorted(os.listdir(location)):
        if files.endswith(".csv") and not files.endswith("13.csv") and files!='Combined_IDLead_1.csv':
            df=pd.read_csv(f"{location}{files}")
            test_final=pd.concat([test_final,df],axis=1,ignore_index=True)
            target = test_final.iloc[:,-1]
            test_final.drop(columns=test_final.columns[-1],axis=1,inplace=True)

    return reduce_features(test_final, target, output_dir, transform, pca_dir, model_name, model_type)

def reduce_features(test_final, target, output_dir, transform, pca_dir, model_name, model_type):

    pca_path = pca_dir

    if not os.path.exists(pca_path):
        os.makedirs(pca_path)

    if transform==0 or transform==2:
        pca = PCA(n_components=0.95)  # Retain 95% variance
        x_pca = pca.fit_transform(test_final)
        if model_type == 0:
            pca_model_path = os.path.join(pca_path, f'rf_pca_{model_name}.pkl')
        elif model_type == 1:
            pca_model_path = os.path.join(pca_path, f'dnn_pca_{model_name}.pkl')
        joblib.dump(pca, pca_model_path)
    else:
        if model_type == 0:
            pca_model_path = os.path.join(pca_path, f'rf_pca_{model_name}.pkl')
        elif model_type == 1:
            pca_model_path = os.path.join(pca_path, f'dnn_pca_{model_name}.pkl')
        pca = joblib.load(pca_model_path)
        x_pca = pca.transform(test_final)

    x_pca = pd.DataFrame(x_pca)
    final_result_df = pd.concat([x_pca, target], axis=1)
    final_result_df['target'] = final_result_df.iloc[:, -1]
    final_result_df.drop(columns=[final_result_df.columns[-2]], inplace=True)
    mapping = {'No': 0, 'HB': 1, 'MI': 2, 'PM': 3}
    final_result_df['target'] = final_result_df['target'].map(mapping)
    output_file_path = os.path.join(output_dir, 'pca_final.csv')
    final_result_df.to_csv(output_file_path, index=False)
    return final_result_df