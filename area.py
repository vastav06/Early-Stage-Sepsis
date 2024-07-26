# import cv2
# import numpy as np
# from PIL import Image
# import json
# import os

# json_file_path= "/Users/vastavbharambe/Downloads/cap/jsonfile/area_data.json"
# if os.path.exists(json_file_path):
#     # Read the existing area data and get the last hour index
#     with open(json_file_path, "r") as json_file:
#         area_data_list = json.load(json_file)
#         last_hour_index = area_data_list[-1]["hour"]
# else:
#     # If the file doesn't exist, start with hour index 1
#     area_data_list = []
#     last_hour_index = 0
# if last_hour_index == 12:
#     print("All hours have been processed.")
#     exit()
# # Increment the hour index for the current execution
# hour_index = last_hour_index + 1
# # Load the image
# img = cv2.imread("/Users/vastavbharambe/Downloads/cap/test_images/2.png")








# ## Adding Padding to the image
# # row, col = img.shape[:2]
# # bottom = img[row-2:row, 0:col]
# # mean = cv2.mean(bottom)[0]

# # border_size = 50
# # img = cv2.copyMakeBorder(
# #     img,
# #     top=border_size,
# #     bottom=border_size,
# #     left=border_size,
# #     right=border_size,
# #     borderType=cv2.BORDER_CONSTANT,
# #     value=[0, 0, 0]
# # )
# # cv2.imshow('border', img)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # First, segment the image by reducing colors using K-Means on the Hue channel
# hue = hsv[:,:,0].flatten().reshape(-1, 1).astype(np.float32)
# K_reduce = 5  # Number of clusters to merge colors
# criteria_reduce = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# _, labels_reduce, centers_reduce = cv2.kmeans(hue, K_reduce, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)
# labels_reduce = labels_reduce.reshape(hsv[:,:,0].shape)

# # Second, filter for red region using inRange for HSV
# lower_red = np.array([0, 50, 50])
# upper_red = np.array([10, 255, 255])
# mask_red = cv2.inRange(hsv, lower_red, upper_red)

# # Extract red area with K-Means based on the filtered mask
# masked_hue = np.where(mask_red > 0, hsv[:,:,0], 0).flatten().reshape(-1, 1).astype(np.float32)
# K_extract = 5  # Number of clusters to extract red area
# _, labels_extract, _ = cv2.kmeans(masked_hue, K_extract, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)
# labels_extract = labels_extract.reshape(hsv[:,:,0].shape)
# extracted_red_area = (labels_extract * (180 // K_extract)).astype(np.uint8)

# # Optionally, you can apply Gaussian blur on the extracted red area to smooth the segmentation
# blurred_image = cv2.GaussianBlur(extracted_red_area, (7, 7), 0)

# # Display the results
# # cv2.imshow('Blurred Extracted Red Area', blurred_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# thresh_Value = 20
# _, binary_image = cv2.threshold(blurred_image, thresh_Value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Find contours in the binary image
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the largest contour based on area
# largest_contour = max(contours, key=cv2.contourArea)
# area = cv2.contourArea(largest_contour)

# blurred_image_with_contour = blurred_image.copy()

# # Draw the largest contour on the blurred image copy
# cv2.drawContours(blurred_image_with_contour, [largest_contour], -1, (255, 255, 255), 2)
# # Draw the area value on the blurred image copy

# font = cv2.FONT_HERSHEY_SIMPLEX
# text = f"Area: {area:.2f}"
# font_scale = 0.8  # Smaller font size
# thickness = 4  # Increased thickness for boldness and clarity

# # Calculate text size to adjust positioning
# (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

# # Position the text at the bottom left, with a margin from the bottom
# x = 10  # Margin from the left edge of the image
# y = blurred_image_with_contour.shape[0] - 10  # Adjusted to ensure text is well within the bottom margin

# # Text color - white for better visibility
# text_color = (255, 255, 255)  # White for high contrast

# # Draw the text on the blurred image
# cv2.putText(blurred_image_with_contour, text, (x, y), font, font_scale, text_color, thickness)

# # Display the final image
# # cv2.imshow('Largest White Area Outlined', blurred_image_with_contour)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# cv2.imwrite("/Users/vastavbharambe/Downloads/cap/saved_images/hour1.png", blurred_image_with_contour)
# blurred_image_with_contour_list = blurred_image_with_contour.tolist()
# # Store the area in a JSON file
# area_data_list.append({"hour": hour_index,"border":blurred_image_with_contour_list, "area": float(area)})
# with open(json_file_path, "w") as json_file:
#     json.dump(area_data_list, json_file)

#Code starts here
import cv2
import numpy as np
from PIL import Image
import json
import os
import psycopg2
def read_area_data(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            area_data_list = json.load(json_file)
            last_hour_index = area_data_list[-1]["hour"]
    else:
        area_data_list = []
        last_hour_index = 0
    return area_data_list, last_hour_index

def process_image(img_path, json_file_path):
    area_data_list, last_hour_index = read_area_data(json_file_path)
    if last_hour_index == 12:
        print("All hours have been processed.")
        return

    hour_index = last_hour_index + 1
    img = cv2.imread(img_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].flatten().reshape(-1, 1).astype(np.float32)
    K_reduce = 5
    criteria_reduce = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_reduce, centers_reduce = cv2.kmeans(hue, K_reduce, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels_reduce = labels_reduce.reshape(hsv[:,:,0].shape)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    masked_hue = np.where(mask_red > 0, hsv[:,:,0], 0).flatten().reshape(-1, 1).astype(np.float32)
    K_extract = 5
    _, labels_extract, _ = cv2.kmeans(masked_hue, K_extract, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels_extract = labels_extract.reshape(hsv[:,:,0].shape)
    extracted_red_area = (labels_extract * (180 // K_extract)).astype(np.uint8)

    blurred_image = cv2.GaussianBlur(extracted_red_area, (7, 7), 0)

    thresh_Value = 20
    _, binary_image = cv2.threshold(blurred_image, thresh_Value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    blurred_image_with_contour = blurred_image.copy()

    cv2.drawContours(blurred_image_with_contour, [largest_contour], -1, (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Area: {area:.2f}"
    font_scale = 0.8
    thickness = 4

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = 10
    y = blurred_image_with_contour.shape[0] - 10
    text_color = (255, 255, 255)

    cv2.putText(blurred_image_with_contour, text, (x, y), font, font_scale, text_color, thickness)
    
    image_bytes = cv2.imencode('.png', blurred_image_with_contour)[1].tobytes()
    
    cv2.imwrite(f"/Users/vastavbharambe/Downloads/cap/saved_images/hour{hour_index}.png", blurred_image_with_contour)
    
    blurred_image_with_contour_list = blurred_image_with_contour.tolist()
    
    area_data_list.append({"hour": hour_index, "border": blurred_image_with_contour_list, "area": float(area)})
    

    with open(json_file_path, "w") as json_file:
        json.dump(area_data_list, json_file)

# Usage example
# img_path = "/Users/vastavbharambe/Downloads/cap/test_images/2.png"
# json_file_path = "/Users/vastavbharambe/Downloads/cap/jsonfile/area_data.json"
# process_image(img_path, json_file_path)










#################################################################################
# ## Selecting the dark gray region
# gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# # Create a mask for the largest contour
# mask_largest_contour = np.zeros_like(gray)  # Now 'gray' is defined
# cv2.drawContours(mask_largest_contour, [largest_contour], -1, color=255, thickness=cv2.FILLED)

# # Apply this mask to the grayscale image
# masked_area = cv2.bitwise_and(gray, gray, mask=mask_largest_contour)

# # Use adaptive thresholding to isolate the dark gray region within the masked area
# # The parameters here may need to be adjusted for your specific image
# adaptive_thresh = cv2.adaptiveThreshold(masked_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# # Find contours in the adaptive threshold image
# contours_gray, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the largest contour among the gray regions
# largest_gray_contour = max(contours_gray, key=cv2.contourArea)

# # Draw the largest gray contour on the original image to outline the dark gray region
# cv2.drawContours(blurred_image, [largest_gray_contour], -1, (0, 255, 0), 2)

# # Display the result
# cv2.imshow('Dark Gray Area Outlined', blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
################################################################################

# # Convert BGR to HSV
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Flatten the Hue channel
# hue = hsv[:,:,0].flatten().reshape(-1, 1).astype(np.float32)

# # Apply K-Means to reduce colors
# K_reduce = 5  # Number of clusters to merge colors
# criteria_reduce = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# _, labels_reduce, centers_reduce = cv2.kmeans(hue, K_reduce, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Reshape labels back to image shape
# labels_reduce = labels_reduce.reshape(hsv[:,:,0].shape)

# # Filter for red region
# lower_red = np.array([0, 50, 50])
# upper_red = np.array([10, 255, 255])
# mask_red = cv2.inRange(hsv, lower_red, upper_red)

# # Apply additional filtering if needed

# # Extract red area with K-Means
# masked_hue = np.where(mask_red > 0, hsv[:,:,0], 0).flatten().reshape(-1, 1).astype(np.float32)
# K_extract = 5  # Number of clusters to extract red area
# _, labels_extract, _ = cv2.kmeans(masked_hue, K_extract, None, criteria_reduce, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Reshape labels back to image shape
# labels_extract = labels_extract.reshape(hsv[:,:,0].shape)
# extracted_red_area = (labels_extract * (180 // K_extract)).astype(np.uint8)

# # Display the results
# cv2.imshow('Extracted Red Area', extracted_red_area)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



################################################################################

# pixel_values = extracted_red_area.reshape((-1, 3))
# # Convert to float
# pixel_values = np.float32(pixel_values)

# # Define criteria and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# # Number of clusters (K)
# k = 3
# _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Convert back to 8 bit values
# centers = np.uint8(centers)

# # Map labels to center values
# segmented_image = centers[labels.flatten()]

# # Reshape back to the original image dimensions
# segmented_image = segmented_image.reshape(extracted_red_area.shape)

# # Show the image
# cv2.imshow('Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()