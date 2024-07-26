import cv2
import numpy as np
from PIL import Image

def extract_exact_rectangle_area(image_array, threshold):
    # Find contours on the thresholded image
    contours, _ = cv2.findContours((image_array > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the minimum area rectangle that bounds the detected contour
        rectangle_contour = max(contours, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(rectangle_contour)
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)

        # Create a mask and calculate the exact area of the rotated rectangle
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
        exact_rectangle_area = np.sum(mask > threshold)
        
        return exact_rectangle_area, box
    else:
        return 0, np.array([])

def calculate_areas_and_ratio(image_paths, threshold):
    images = [np.array(Image.open(path).convert('L')) for path in image_paths]
    
    areas = []
    boxes = []
    visualized_images = []
    for img in images:
        area, box = extract_exact_rectangle_area(img, threshold)
        areas.append(area)
        boxes.append(box)

        # Visualize the rotated rectangle on the original image
        visualized_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if box.size != 0:
            cv2.drawContours(visualized_image, [box], 0, (0, 255, 0), 2)
        visualized_images.append(visualized_image)
    
    ratio = areas[1] / areas[0] if areas[0] != 0 else 0

    return areas, np.round(ratio,1), boxes, visualized_images

# Paths to your images
image_path1 = '/Users/dev/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Assignments/Capstone/Code/test_images/abc.jpeg'
image_path2 = '/Users/dev/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Assignments/Capstone/Code/test_images/abcd.jpeg'
image_paths = [image_path1, image_path2]

# Threshold for detecting white areas
threshold = 200

# Calculate areas, ratio, and bounding boxes
areas, area_ratio, boxes, visualized_images = calculate_areas_and_ratio(image_paths, threshold)

for i, vis_img in enumerate(visualized_images):
    cv2.imshow(f"Image {i+1}", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"Area of rectangle 1: {areas[0]} pixels")
print(f"Area of rectangle 2: {areas[1]} pixels")
print(f"Ratio of areas: {area_ratio}")