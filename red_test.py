import cv2
import numpy as np

def find_and_outline_red_object(cap):
    delay = 50  # Delay in milliseconds
    area_threshold = 50  # Threshold for area change (adjust as needed)
    prev_area = 0

    while True:
        _, frame = cap.read()

        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour and draw it
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if the area change is significant
            if abs(area - prev_area) > area_threshold:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                cv2.putText(frame, f"Area: {area:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prev_area = area

        # Display the frame
        cv2.imshow('frame', frame)

        key = cv2.waitKey(delay)
        if key & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Use the primary camera
    find_and_outline_red_object(cap)
    cap.release()
    cv2.destroyAllWindows()