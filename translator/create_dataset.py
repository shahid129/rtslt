import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# all working

# IMAGE_DIR = "data"
# OUTPUT_CSV = "dataset.csv"

# project root directory
# Go one level up
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# image directory and output CSV
IMAGE_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "dataset.csv")

# Initialize hand for mediapipe
mp_hands = mp.solutions.hands
hand_processor = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

dataset = []
labels = []


def process_image_directory(directory):
    # process each folder in the directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue

        for image_file in os.listdir(folder_path):
            # process each image file in the folder
            process_image_file(folder, folder_path, image_file)


def process_image_file(label, folder_path, image_file):
    # read the image file from the folder
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image {image_file}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect hand landmarks in the image
    results = hand_processor.process(image_rgb)
    if results.multi_hand_landmarks:
        extract_landmarks(label, results.multi_hand_landmarks)


def extract_landmarks(label, hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        # List to store x and y-coordinates of the landmarks
        x_coords = []
        y_coords = []

        # store normalized landmarks data
        landmarks_data = []

        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            landmarks_data.append(landmark.x - min(x_coords))
            landmarks_data.append(landmark.y - min(y_coords))

        if len(landmarks_data) == 42:
            # append the label as the last element
            landmarks_data.append(int(label))
            # add the data point to the dataset
            dataset.append(landmarks_data)

#  process and save the images
process_image_directory(IMAGE_DIR)

# convert to DataFrame and  save to dataset.csv
dataframe = pd.DataFrame(dataset)

# save the data to a CSV file
dataframe.to_csv(OUTPUT_CSV, index=False, header=False)

print(f"Dataset created and saved to {OUTPUT_CSV}")
