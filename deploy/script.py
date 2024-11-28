import os
import time
from datetime import datetime

import boto3
import cv2
from ultralytics import YOLO

# Load environment variables
try:
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.environ.get("AWS_REGION")
    BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    exit()

# Initialize YOLO model
try:
    model = YOLO("models/best.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Initialize webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit()

# Configure AWS credentials
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
except Exception as e:
    print(f"Error connecting to AWS: {e}")
    exit()


# Function for processing image
def process_frame():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame from camera")
        return None, None

    # Run YOLO detection
    results = model(frame)

    # Get the number of detected cows
    num_cows = len(results[0].boxes)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    return annotated_frame, num_cows


# Function for uploading image to S3
def upload_to_s3(image, num_cows):
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save image temporarily
    temp_path = f"temp_{timestamp}.jpg"
    try:
        cv2.imwrite(temp_path, image)
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError("Generated image file is empty")
        # Define S3 path based on number of cows
        s3_path = f"{num_cows}/{timestamp}.jpg"

        # Upload to S3
        try:
            s3_client.upload_file(
                temp_path, BUCKET_NAME, s3_path, ExtraArgs={"ContentType": "image/jpeg"}
            )
            print(f"Uploaded image with {num_cows} cows to {s3_path}")
        except Exception as e:
            print(f"Error uploading to S3: {e}")
        finally:
            # Clean up temporary file
            os.remove(temp_path)
    except Exception as e:
        print(f"Error saving image: {e}")


# Main function
def main():
    while True:
        # Process frame
        frame, num_cows = process_frame()
        if frame is None:
            continue

        # Upload to S3
        upload_to_s3(frame, num_cows)

        # Wait for 5 seconds before next capture
        time.sleep(8)


# Run main function
if __name__ == "__main__":
    try:
        main()
    finally:
        cap.release()