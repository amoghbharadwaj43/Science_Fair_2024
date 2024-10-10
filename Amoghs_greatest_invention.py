import cv2
from datetime import datetime
import openpyxl
import os
import geocoder

# Static distance between two cameras in inches (64 inches)
distance_between_cameras_inches = 64.0
# Convert distance to meters (1 inch = 0.0254 meters)
distance_between_cameras_meters = distance_between_cameras_inches * 0.0254

# Static values
ACTUAL_SPEED = 0.79  # mph

# Load car classifier
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Initialize the two cameras
camera_1 = cv2.VideoCapture(0)  # Camera 1
camera_2 = cv2.VideoCapture(1)  # Camera 2

if not camera_1.isOpened() or not camera_2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

file_path = "car_speed_data.xlsx"
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active


# Function to detect car and get timestamp
def detect_car(camera, window_name, scale_factor=1.1, min_neighbors=2):
    while True:
        ret, frame = camera.read()

        if not ret:
            print(f"Error: Could not read frame from {window_name}.")
            break

        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(frame, scale_factor, min_neighbors)

        # If a car is detected, return the current timestamp
        if len(cars) > 0:
            timestamp = datetime.now()
            for (x, y, w, h) in cars:
                # Draw rectangle around the detected car
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Car Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Print message when car is detected
            print(f"{window_name}: Car detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            cv2.imshow(window_name, frame)
            return timestamp

        # Display the camera feed
        cv2.imshow(window_name, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Function to get laptop's current location (latitude and longitude)
def get_lat_lon():
    g = geocoder.ip('me')  # You can change to 'gps' if actual GPS is available
    return g.latlng if g.latlng else (None, None)


# Main loop to detect car in both cameras
try:
    print("Waiting for car to pass Camera 1...")
    timestamp_1 = detect_car(camera_1, "Camera 1", scale_factor=1.05, min_neighbors=4)
    print(f"Car detected at Camera 1 at {timestamp_1}")

    # Wait for the car to reach Camera 2 and detect it
    print("Waiting for car to pass Camera 2...")
    timestamp_2 = None
    while timestamp_2 is None:
        timestamp_2 = detect_car(camera_2, "Camera 2", scale_factor=1.05, min_neighbors=4)
    print(f"Car detected at Camera 2 at {timestamp_2}")

    # Calculate time difference between two detections
    time_diff = (timestamp_2 - timestamp_1).total_seconds()

    # Calculate speed using the formula speed = distance / time
    if time_diff > 0:
        speed_mps = distance_between_cameras_meters / time_diff  # Speed in m/s
        speed_mph = speed_mps * 2.23694  # Convert m/s to mph
        print(f"Car speed: {speed_mph:.2f} mph")

        # Get laptop's latitude and longitude
        lat, lon = get_lat_lon()

        # Calculate accuracy
        accuracy = (ACTUAL_SPEED - (ACTUAL_SPEED - speed_mph) / ACTUAL_SPEED) * 100
        if accuracy > 100:
            accuracy -=100
        # Append the date, speed, location, actual speed, and accuracy to the Excel sheet
        sheet.append([timestamp_1.strftime('%Y-%m-%d %H:%M:%S'),
                      f"{speed_mph:.2f}",
                      lat, lon,
                      ACTUAL_SPEED,
                      f"{accuracy:.2f}"])

        # Save the workbook
        workbook.save(file_path)
    else:
        print("Error: Time difference is zero or negative.")

finally:
    # Release both cameras and close all windows
    camera_1.release()
    camera_2.release()
    cv2.destroyAllWindows()
