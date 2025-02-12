import cv2 as cv
import numpy as np
import os
from deepface import DeepFace
from retinaface import RetinaFace
import threading
from TimeClockMarker import save_recognized_face
import time

liveCap = cv.VideoCapture(0)  # Initialize camera

if not liveCap.isOpened():
    print("Cannot open camera")
    exit()

# Global variables for threading
latest_frame = None
latest_result = None
latest_landmarks = None
lock = threading.Lock()

# Global variables for the virtual line
line_y = 150  # Y-coordinate of the virtual line (horizontal line)
line_start_x = 20  # Start X-coordinate of the line
line_end_x = 90  # End X-coordinate of the line
triggered = False  # Flag to avoid repeated recognition

# Confidence threshold for recognition
confidence_threshold = 0.75

# Cooldown period to avoid duplicate recognitions
last_recognition_time = 0
cooldown_period = 5  # 5 seconds

def process_frame():
    global latest_frame, latest_result, latest_landmarks, lock, triggered, last_recognition_time
    while True:
        with lock:
            if latest_frame is None:
                continue
            img = latest_frame.copy()

        # Detect faces and landmarks
        faces = RetinaFace.detect_faces(img, threshold=0.5)

        if not faces:
            print("No face detected.")
            latest_result = None
            latest_landmarks = None
            triggered = False  # Reset the trigger if no face is detected
        else:
            try:
                for face_id, face_data in faces.items():
                    facial_area = face_data["facial_area"]
                    x, y, w, h = facial_area

                    face_center_y = y + h // 2  # Calculate the center of the face's bounding box

                    if face_center_y < line_y and not triggered and time.time() - last_recognition_time > cooldown_period:
                        print("Face crossed the line! Recognizing...")
                        triggered = True  # Set the trigger to avoid repeated recognition

                        # Facial recognition
                        dfs = DeepFace.find(img_path=img,
                                            db_path="dataset",
                                            model_name="Facenet",
                                            enforce_detection=False)

                        # Check for valid results
                        if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                            first_match = dfs[0].iloc[0]
                            identity_path = first_match["identity"]

                            # Normalize path for compatibility
                            identity_parts = os.path.normpath(identity_path).split(os.sep)

                            # Avoid index error
                            if len(identity_parts) >= 2:
                                name = identity_parts[-2]  # Folder name inside "dataset"
                            else:
                                name = "Unknown"

                            confidence = 1 - first_match["distance"]
                            if confidence >= confidence_threshold:
                                print(f"Recognized: {name} (Confidence: {confidence:.2f})")
                                save_recognized_face(name, img)
                                last_recognition_time = time.time()
                            else:
                                name = "Unknown"
                                print(f"Low confidence match: {confidence:.2f}")

                            landmarks = face_data["landmarks"]  # Extract landmarks from the detected face

                            with lock:
                                latest_result = (name, confidence)
                                latest_landmarks = landmarks
                        else:
                            print("No matching face found")
                            latest_result = None
                            latest_landmarks = None
                    elif face_center_y >= line_y:
                        triggered = False  # Reset the trigger if the face moves back below the line

            except ValueError:
                print("No face detected")
                latest_result = None
                latest_landmarks = None

# Start the processing thread
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

frame_counter = 0
skip_frames = 1  # Process every x frame

while True:
    ret, frame = liveCap.read()

    if not ret:
        print("Failed to capture frame")
        continue

    frame = cv.resize(frame, (120, 160))  # Downscale the frame for faster processing

    # Update the latest frame
    with lock:
        latest_frame = frame

    # Skip frames to increase frequency
    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    cv.line(frame, (line_start_x, line_y), (line_end_x, line_y), (0, 255, 0), 2)  # Draw the shortened virtual line on the frame

    # Display the result and landmarks
    with lock:
        if latest_result is not None and latest_landmarks is not None:
            name, confidence = latest_result
            landmarks = latest_landmarks

            # Draw the name and confidence on the frame
            cv.putText(frame, name, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw facial landmarks (dots on the face)
            for landmark_name, (x, y) in landmarks.items():
                cv.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red dots

    cv.imshow("Face Detection", frame)  # Show the frame

    # Exit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
liveCap.release()
cv.destroyAllWindows()