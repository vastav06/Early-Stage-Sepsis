# def zoom_out_rectangle(rect, scale_factor, center):
#     # Calculate the center of the rectangle
#     rect_center = ((rect[0][0] + rect[2][0]) / 2, (rect[0][1] + rect[2][1]) / 2)
    
#     # Calculate the distance from the center to each corner
#     dx1, dy1 = rect[0][0] - rect_center[0], rect[0][1] - rect_center[1]
#     dx2, dy2 = rect[2][0] - rect_center[0], rect[2][1] - rect_center[1]
    
#     # Scale the distances
#     dx1_scaled, dy1_scaled = dx1 * scale_factor, dy1 * scale_factor
#     dx2_scaled, dy2_scaled = dx2 * scale_factor, dy2 * scale_factor
    
#     # Calculate the new corners of the rectangle
#     new_rect = [(int(rect_center[0] + dx1_scaled), int(rect_center[1] + dy1_scaled)),
#                 (int(rect_center[0] + dx2_scaled), int(rect_center[1] + dy1_scaled)),
#                 (int(rect_center[0] + dx2_scaled), int(rect_center[1] + dy2_scaled)),
#                 (int(rect_center[0] + dx1_scaled), int(rect_center[1] + dy2_scaled))]
    
#     return new_rect

# # Define the original rectangle points
# rect_points = [(279, 1153), (460, 1060), (529, 1194), (348, 1286)]

# # Define the scale factor
# scale_factor = 0.9254032445

# # Calculate the center of the rectangle
# rect_center = ((rect_points[0][0] + rect_points[2][0]) / 2, (rect_points[0][1] + rect_points[2][1]) / 2)

# # Zoom out the rectangle
# new_rect_points = zoom_out_rectangle(rect_points, scale_factor, rect_center)

# # Print the new rectangle points
# print("New rectangle points after zooming out:")
# for point in new_rect_points:
#     print("-", point)







# import cv2
# def zoom_at(img, zoom=1, angle=0, coord=None):
    
#     cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
#     rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
#     result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
#     return result
# img = cv2.imread('/Users/vastavbharambe/Downloads/cap/final_images/hour3/h3nw.jpeg')
# img2 = zoom_at(img, 0.9254032445, coord=(362.8962097167969, 1155.88671875))
# print(img.shape)
# print(img2.shape)
# cv2.imwrite('x3Lkg_zoomed.png', img2 )



import cv2
import numpy as np

def rotate_point(point, angle, center):
    angle_rad = np.radians(angle)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    x = center[0] + (point[0] - center[0]) * cos_val - (point[1] - center[1]) * sin_val
    y = center[1] + (point[0] - center[0]) * sin_val + (point[1] - center[1]) * cos_val
    return (int(x), int(y))

# Define the original rectangle points
original_rect_points = [(288, 1154), (519, 1154), (519, 1192), (288, 1192)]

# Define the target rectangle points
target_rect_points = [(773, 561), (792, 354), (945, 368), (926, 575)]

# Calculate the center of each rectangle
original_rect_center = ((original_rect_points[0][0] + original_rect_points[2][0]) // 2, (original_rect_points[0][1] + original_rect_points[2][1]) // 2)
target_rect_center = ((target_rect_points[0][0] + target_rect_points[2][0]) // 2, (target_rect_points[0][1] + target_rect_points[2][1]) // 2)

# Calculate the translation required to move the original rectangle to align with the target rectangle
translation = (target_rect_center[0] - original_rect_center[0], target_rect_center[1] - original_rect_center[1])

# Calculate the angle of rotation to match the orientation of the target rectangle
original_angle = np.degrees(np.arctan2(original_rect_points[1][1] - original_rect_points[0][1], original_rect_points[1][0] - original_rect_points[0][0]))
target_angle = np.degrees(np.arctan2(target_rect_points[1][1] - target_rect_points[0][1], target_rect_points[1][0] - target_rect_points[0][0]))
rotation_angle = target_angle - original_angle

# Rotate and translate the original rectangle points to match the target rectangle
rotated_rect_points = [rotate_point(point, rotation_angle, original_rect_center) for point in original_rect_points]
translated_rotated_rect_points = [(point[0] + translation[0], point[1] + translation[1]) for point in rotated_rect_points]

# Define the path to the image file
image_path = '/Users/vastavbharambe/Downloads/cap/x3Lkg_zoomed.png'

# Read the image
image = cv2.imread(image_path)

# Rotate and translate the entire image
rot_mat = cv2.getRotationMatrix2D(original_rect_center, rotation_angle, 1.0)
translated_rotated_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
translated_rotated_image = cv2.warpAffine(translated_rotated_image, np.float32([[1, 0, translation[0]], [0, 1, translation[1]]]), (image.shape[1], image.shape[0]))

# Display the rotated and translated image
cv2.imshow('Rotated and Translated Image', translated_rotated_image)
cv2.imwrite('Rotated_and_Translated_Image.png', translated_rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
