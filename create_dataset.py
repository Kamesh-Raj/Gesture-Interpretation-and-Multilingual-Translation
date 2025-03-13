import cv2
import numpy as np
import os
import time
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize cvzone HandDetector
hand_detector = HandDetector(detectionCon=0.5, maxHands=2)

# Load labels from labels.txt
with open('labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Path setup
base_path = 'C:\\gesture_interpretation1\\sld10\\'
for label in labels:
    os.makedirs(os.path.join(base_path, label), exist_ok=True)

# Capture settings
cap = cv2.VideoCapture(0)
sequence_length = 30  # Number of frames per sequence
total_sequences = 30  # Total sequences to capture
num_landmarks = 42  # 21 landmarks * 2 hands

def calculate_movement(curr_hand_landmarks, prev_hand_landmarks):
    hand_movement = np.zeros(2)
    if prev_hand_landmarks is not None:
        # Calculate the average position for both hands
        curr_avg_x = np.mean(curr_hand_landmarks[:, 0])
        curr_avg_y = np.mean(curr_hand_landmarks[:, 1])
        prev_avg_x = np.mean(prev_hand_landmarks[:, 0])
        prev_avg_y = np.mean(prev_hand_landmarks[:, 1])
        
        # Calculate the movement direction
        if curr_avg_y > prev_avg_y:
            hand_movement[1] = 1  # Up to Down
        else:
            hand_movement[1] = -1  # Down to Up

        if curr_avg_x > prev_avg_x:
            hand_movement[0] = 1  # Right to Left
        else:
            hand_movement[0] = -1  # Left to Right
    return hand_movement

# Start the capturing process for each label
for label in labels:
    print(f"Capturing sequences for label: {label}")
    sequence_count = 0
    
    while sequence_count < total_sequences:  # Capture total_sequences for each label
        os.makedirs(os.path.join(base_path, label, f"{label}_seq{sequence_count}"), exist_ok=True)
        frames = []  # Store frames for each sequence
        features = []  # Store features for each sequence
        prev_hand_landmarks = None  # Previous landmarks for hand movement calculation
        
        while len(frames) < sequence_length:  # Capture frames until the sequence length is reached
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect hands using cvzone HandDetector
            hands, img = hand_detector.findHands(frame, flipType=True)
            if hands:  # Proceed only if hands are detected
                # Convert the BGR image to RGB and process it with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract hand landmarks
                hand_landmarks = []
                for hand in hands:
                    lm_list = hand["lmList"]  # Get landmark list for the hand
                    hand_landmarks.append(np.array(lm_list))  # Add hand landmarks to the list
                
                # Mirror the landmarks if only right hand is used
                if len(hand_landmarks) == 1 and hand_landmarks[0].shape[0] == 21:
                    mirrored_landmarks = np.copy(hand_landmarks[0])
                    mirrored_landmarks[:, 0] = frame.shape[1] - mirrored_landmarks[:, 0]  # Mirror X coordinates
                    hand_landmarks.append(mirrored_landmarks)
                
                # Flatten the landmarks into a single array
                curr_hand_landmarks = np.array(hand_landmarks).flatten()
                # If less than the expected number of landmarks, pad with zeros
                if len(curr_hand_landmarks) < num_landmarks:
                    curr_hand_landmarks = np.pad(curr_hand_landmarks, (0, num_landmarks - len(curr_hand_landmarks)), mode='constant')
                else:
                    curr_hand_landmarks = curr_hand_landmarks[:num_landmarks]  # Truncate if needed
                
                # Calculate hand movement
                hand_movement = calculate_movement(np.array(hand_landmarks), prev_hand_landmarks)
                
                combined_features = np.concatenate((curr_hand_landmarks, hand_movement))
                features.append(combined_features)
                frames.append(frame)
                prev_hand_landmarks = np.array(hand_landmarks)
                
                # Display current label and sequence number on the frame
                cv2.putText(frame, f"Capturing: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Sequence: {sequence_count + 1}/{total_sequences}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Capture Gesture Sequence', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save each frame and feature of the sequence only if they are not empty
        if frames and features:  # Ensure there are frames and features to save
            try:
                np.save(os.path.join(base_path, label, f"{label}_seq{sequence_count}", f"{label}_seq{sequence_count}_frames.npy"), np.array(frames))
                np.save(os.path.join(base_path, label, f"{label}_seq{sequence_count}", f"{label}_seq{sequence_count}_features.npy"), np.array(features))
                sequence_count += 1
                print(f"Captured sequence {sequence_count}/{total_sequences} for label: {label}")
                time.sleep(1)  # Small pause between sequences
            except Exception as e:
                print(f"Error saving sequence {sequence_count} for label {label}: {e}")

    print(f"Completed capturing for label: {label}")

cap.release()
cv2.destroyAllWindows()
print("Dataset capture completed.")