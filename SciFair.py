import cv2
from datetime import datetime
import openpyxl
import geocoder
import numpy as np
from datetime import timedelta
# Static distance between two cameras in meters (1 inch = 0.0254 meters, 64 inches)
DISTANCE_BETWEEN_CAMERAS_METERS = 64.0 * 0.0254
# Static actual speed for comparison (mph)
ACTUAL_SPEED = 0.79

# Load car classifier
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Initialize the two cameras
camera_1 = cv2.VideoCapture(0)  # Camera 1
camera_2 = cv2.VideoCapture(1)  # Camera 2

if not camera_1.isOpened() or not camera_2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Excel file path
file_path = "car_speed_data.xlsx"
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active


# Function to detect cars and return timestamp, frame, and car coordinates
def detect_car(camera, window_name, scale_factor=1.1, min_neighbors=2):
    ret, frame = camera.read()

    if not ret:
        print(f"Error: Could not read frame from {window_name}.")
        return None, None, None

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(frame, scale_factor, min_neighbors)

    if len(cars) > 0:
        timestamp = datetime.now()
        for (x, y, w, h) in cars:
            # Draw rectangle around the detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, 'Car Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)
        return timestamp, frame, cars[0]

    # Display the camera feed
    cv2.imshow(window_name, frame)
    return None, None, None


# Function to detect the primary color of the car
def detect_primary_color(frame, car_coordinates):
    (x, y, w, h) = car_coordinates
    car_image = frame[y:y + h, x:x + w]

    # Resize for faster processing
    resized_image = cv2.resize(car_image, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = np.float32(resized_image.reshape(-1, 3))

    # Use k-means clustering to detect the most dominant color
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant_color = palette[0].astype(int)
    return dominant_color


# Function to get laptop's current location
def get_lat_lon():
    g = geocoder.ip('me')
    return g.latlng if g.latlng else (None, None)


# Function to calculate speed
def calculate_speed(time_diff):
    if time_diff > 0:
        speed_mps = DISTANCE_BETWEEN_CAMERAS_METERS / time_diff
        return speed_mps * 2.23694  # Convert m/s to mph
    return 0


# Function to calculate accuracy based on actual speed
def calculate_accuracy(detected_speed):
    if detected_speed > 0:
        accuracy = (1 - (abs(ACTUAL_SPEED - detected_speed) / ACTUAL_SPEED)) * 100
        return min(accuracy, 100)
    return 0


# Main detection loop
timestamp_1 = None
timestamp_2 = None
camera_1_frame = None
camera_2_frame = None
car_1_coords = None
car_2_coords = None

try:
    while True:
        # Detect car in Camera 1
        if timestamp_1 is None:
            timestamp_1, camera_1_frame, car_1_coords = detect_car(camera_1, "Camera 1", scale_factor=1.22,
                                                                   min_neighbors=4)

        # Detect car in Camera 2
        if timestamp_2 is None:
            timestamp_2, camera_2_frame, car_2_coords = detect_car(camera_2, "Camera 2", scale_factor=1.14,
                                                                   min_neighbors=5)
        # Check if one camera detected the car but the other did not within 8 seconds
        if (timestamp_1 and not timestamp_2 and (datetime.now() - timestamp_1).total_seconds() >= 8) or \
                (timestamp_2 and not timestamp_1 and (datetime.now() - timestamp_2).total_seconds() >= 8):
            # Insert a null row into the Excel file
            sheet.append([timestamp_1.strftime('%Y-%m-%d %H:%M:%S') if timestamp_1 else "null",
                          "null", "null", "null", "null", "null", "null", "null"])
            workbook.save(file_path)

            # Reset timestamps
            timestamp_1 = None
            timestamp_2 = None

        # Both cameras have detected the car, calculate speed
        if timestamp_1 and timestamp_2:
            # Calculate time difference
            time_diff = abs((timestamp_2 - timestamp_1).total_seconds())
            # Subtract 3 seconds from both timestamps if both are available
            if timestamp_1 and timestamp_2:
                # Subtract 3 seconds from both timestamps
                timestamp_1 -= timedelta(seconds=3)
                timestamp_2 -= timedelta(seconds=3)

                # Calculate time difference after adjustment
                time_diff = abs((timestamp_2 - timestamp_1).total_seconds())

            # Adjust time difference if Camera 2 detected first
            if timestamp_2 < timestamp_1:
                print("Camera 2 detected first. Subtracting 3 seconds from time difference.")
                time_diff = max(0, time_diff - 3)

            # Calculate speed and accuracy
            speed_mph = calculate_speed(time_diff)
            accuracy = calculate_accuracy(speed_mph)

            # Detect car's color
            dominant_color = detect_primary_color(camera_1_frame if timestamp_1 < timestamp_2 else camera_2_frame,
                                                  car_1_coords if timestamp_1 < timestamp_2 else car_2_coords)

            # Get location
            lat, lon = get_lat_lon()

            # Log the data
            sheet.append([timestamp_1.strftime('%Y-%m-%d %H:%M:%S'),
                          f"{speed_mph:.2f}",
                          lat, lon,
                          ACTUAL_SPEED,
                          f"{accuracy:.2f}",
                          f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})",
                          1 if timestamp_1 < timestamp_2 else 2])

            workbook.save(file_path)

            # Reset timestamps
            timestamp_1 = None
            timestamp_2 = None

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    camera_1.release()
    camera_2.release()
    cv2.destroyAllWindows()

