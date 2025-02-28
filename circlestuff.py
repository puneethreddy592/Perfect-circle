import cv2
import numpy as np
import time

def int_to_string(number):
    return str(number)

def find_red_pixels(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range of red color in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Threshold the image to get only red colors
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # Combine the masks
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Find the coordinates of the red pixels
    red_pixel_coordinates = np.column_stack(np.where(red_mask > 0))
    
    # Convert the coordinates to a list of tuples
    red_pixel_coordinates_list = [(x, y) for y, x in red_pixel_coordinates]
    
    return red_pixel_coordinates_list

def count_white_pixels(coordinates, image):
    white_count = 0
    
    # Loop through each coordinate in the list
    for (x, y) in coordinates:
        # Get the pixel value at the coordinate (BGR format)
        pixel_value = image[y, x]
        
        # Check if the pixel is white (255, 255, 255 in BGR)
        if all(pixel_value == [255, 255, 255]):
            white_count += 1
            
    return white_count

def main():
    # Variables to store and show the detected circle properties
    radius_str = ""
    xcenter_str = ""
    ycenter_str = ""
    rvalue = 0
    xvalue = 0
    yvalue = 0
    
    # Variables to store the webcam video and a converted version of the video
    colored_image = None
    gray_image = None
    
    # Auxiliary variable to quit the loop and end the program
    key = 0
    
    # Open the default camera
    capture = cv2.VideoCapture(0)
    
    # # Check for failure
    if not capture.isOpened():
        print("Failed to open the webcam")
        return
    
    # Set capture device properties
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables to track time and max accuracy
    start_time = time.time()
    max_accuracy = 0
    
    # Loop will stop after 8 seconds or if "q" is pressed on the keyboard
    while key != ord('q'):
        # Capture a frame of the webcam live video and store it on the image variable
        ret, colored_image = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        gray_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
        praful_image = np.zeros_like(colored_image)
       
        edges = cv2.Canny(colored_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contour_image = np.zeros_like(colored_image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        
        # Create a vector to store the center value (x and y coordinates) and the radius of each detected circle
        circles = cv2.HoughCircles(
            gray_image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=300,          # Increase minDist to avoid detecting multiple circles too close
            param1=171,           # Adjust the higher threshold for Canny edge detector
            param2=35,            # Lower threshold for center detection
            minRadius=53,         # Minimum radius to detect
            maxRadius=300         # Maximum radius to detect
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Get the center coordinates and radius of the detected circle
                center = (i[0], i[1])
                radius = i[2]
                
                # Store these values into variables to be converted into string and displayed on the image
                rvalue = radius
                xvalue = i[0]
                yvalue = i[1]
                
                # DRAWING THE CENTER OF THE CIRCLE
                cv2.circle(colored_image, center, 3, (0, 255, 0), 2)
                
                # DRAWING THE CIRCLE CONTOUR
                cv2.circle(colored_image, center, radius, (0, 0, 255), 2)
                cv2.circle(praful_image, center, radius, (0 ,0 ,255), 2)
                # Convert the integer center point and radius values to string
                radius_str = int_to_string(rvalue)
                xcenter_str = int_to_string(xvalue)
                ycenter_str = int_to_string(yvalue)
               
                red_list = find_red_pixels(praful_image)
                correct_points = count_white_pixels(red_list, contour_image)
                total_points = len(red_list)
                if total_points > 0:  # Avoid division by zero
                    accuracy = (correct_points / total_points) * 100
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                    cv2.putText(colored_image, f"Accuracy: {accuracy:.2f}%", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # Display the center and radius values
                cv2.putText(colored_image, f"({xcenter_str}, {ycenter_str})", (xvalue, yvalue - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        else:
            print("No circles detected")

        # Show your results
        cv2.imshow("Hough Circle Transform Demo", colored_image)
        # cv2.imshow("praful image", praful_image)
        # cv2.imshow('Contours', contour_image)

        # Check for user input to quit the loop
        key = cv2.waitKey(25)

        # Stop the loop after 8 seconds
        if time.time() - start_time > 8:
            break
    
    # After the loop, display the max accuracy
    final_image = colored_image.copy()
    cv2.putText(final_image, f"Max Accuracy: {max_accuracy:.2f}%", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Final Result with Max Accuracy", final_image)
    cv2.waitKey(0)

    # Release the capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()