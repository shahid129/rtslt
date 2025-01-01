import os

import cv2

# Directory to save the collected images
IMAGE_DIR = "./data"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Number of classes and images per class
NUM_CLASSES = 25
IMAGES_PER_CLASS = 100

# Initialize video capture
# For my laptop, the webcam is at index 1.
# For other laptops, change the index to 0.
camera = cv2.VideoCapture(1)


def create_class_directory(class_id):
    class_dir = os.path.join(IMAGE_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    return class_dir


def collect_images_for_class(class_id):
    class_dir = create_class_directory(class_id)
    print(f"Collecting data for class {class_id}")

    # Wait for user to be ready
    while True:
        ret, frame = camera.read()
        cv2.putText(
            frame,
            'To Start Press "Q"',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    # Collect images for the current class
    image_count = 0
    while image_count < IMAGES_PER_CLASS:
        ret, frame = camera.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f"{image_count}.jpg"), frame)
        image_count += 1


# Collect images for each class
for class_id in range(NUM_CLASSES):
    collect_images_for_class(class_id)

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
