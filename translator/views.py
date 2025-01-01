import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render


# render the main page
def index(request):
    return render(request, "index.html")


# load the trained model from the pickle file
with open("model.p", "rb") as model_file:
    model_data = pickle.load(model_file)
hand_sign_model = model_data["model"]

# mediapipe modules for hand tracking and drawing on the camera feed
mediapipe_hands = mp.solutions.hands
mediapipe_draw = mp.solutions.drawing_utils
mediapipe_styles = mp.solutions.drawing_styles

# initialize hand detection with mediapipe
hand_detector = mediapipe_hands.Hands(
    static_image_mode=False, min_detection_confidence=0.3
)

# dictionary to map label numbers to hand signs
hand_sign_labels = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y",
    24: "Space",
}

# global variables to manage detected letters and messages
detected_letter = ""
message = ""
# flag to enable/disable message concatenation
concatenate_message = False
# Timestamp when a letter stability starts
stability_start_time = None
stable_letter = None


# generate video frames for the hand detection process
def generate_frames():
    global detected_letter
    global message
    global concatenate_message
    global stability_start_time
    global stable_letter

    # open the webcam
    video_capture = cv2.VideoCapture(
        1
    )  # on my laptop, the webcam is at index 1, 0 is my phone for some reason
    while True:
        hand_data = []
        x_coords = []
        y_coords = []

        # read a frame from the webcam
        success, video_frame = video_capture.read()
        if not success:
            break

        frame_height, frame_width, _ = video_frame.shape
        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

        # process the frame for hand landmarks
        hand_results = hand_detector.process(rgb_frame)
        detected_letter_display = "Not detected"  # default display text

        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                # draw the detected hand landmarks on the frame
                mediapipe_draw.draw_landmarks(
                    video_frame,
                    landmarks,
                    mediapipe_hands.HAND_CONNECTIONS,
                    mediapipe_styles.get_default_hand_landmarks_style(),
                    mediapipe_styles.get_default_hand_connections_style(),
                )

                # collect coordinates of each landmark
                for point in landmarks.landmark:
                    x_coords.append(point.x)
                    y_coords.append(point.y)

                # normalize coordinates relative to the bounding box
                for point in landmarks.landmark:
                    relative_x = point.x - min(x_coords)
                    relative_y = point.y - min(y_coords)
                    hand_data.extend([relative_x, relative_y])

                # make a prediction if we have the required data length
                if len(hand_data) == 42:
                    current_time = time.time()
                    prediction = hand_sign_model.predict([np.asarray(hand_data)])
                    detected_sign = hand_sign_labels[int(prediction[0])]
                    prediction_proba = hand_sign_model.predict_proba(
                        [np.asarray(hand_data)]
                    )
                    accuracy = np.max(prediction_proba) * 100

                    # Check stability for 2 seconds
                    if accuracy >= 40:
                        if stability_start_time is None:
                            stability_start_time = current_time
                        elif (current_time - stability_start_time) >= 2:
                            stable_letter = detected_sign
                            detected_letter = stable_letter
                            # Reset stability timer
                            stability_start_time = None

                            # Add stable letter to the message
                            # if concatenation is enabled
                            if concatenate_message:
                                if stable_letter == "Space":
                                    message += " "
                                else:
                                    message += stable_letter
                    else:
                        stability_start_time = (
                            None  # Reset if the prediction is not stable
                        )

                    detected_letter_display = f"{detected_sign} ({accuracy:.2f}%)"

                    # draw bounding box around detected hand
                    box_x1 = int(min(x_coords) * frame_width) - 10
                    box_y1 = int(min(y_coords) * frame_height) - 10
                    box_x2 = int(max(x_coords) * frame_width) - 10
                    box_y2 = int(max(y_coords) * frame_height) - 10

                    cv2.rectangle(
                        video_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2
                    )
                    cv2.putText(
                        video_frame,
                        detected_letter_display,
                        (box_x1, box_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # encode the frame to send as a stream
        ret, buffer = cv2.imencode(".jpg", video_frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# Return the detected letter as Json response
def get_detected_letter(request):
    global detected_letter
    return JsonResponse({"letter": detected_letter})


# processed video frames as a stream
def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


# concatenate the detected letters into a message as a Json
def get_message(request):
    global message
    return JsonResponse({"message": message})


# reset the message
def reset_message(request):
    global message
    message = ""
    return JsonResponse({"status": "Message reset"})


# endpoint to start concatenating detected letters into a message
def start_message(request):
    global concatenate_message
    concatenate_message = True
    return JsonResponse({"status": "Message concatenation started"})


# stop concatenating detected letters into a message
def stop_message(request):
    global concatenate_message
    concatenate_message = False
    return JsonResponse({"status": "Message concatenation stopped"})
