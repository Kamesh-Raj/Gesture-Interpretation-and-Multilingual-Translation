from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import time
from googletrans import Translator
import mediapipe as mp

app = Flask(__name__)

# Load the trained model
cnn_lstm_model = tf.keras.models.load_model('cnn_lstm_gesture_recognition_model1.h5')
unique_labels = np.load('unique_labels.npy', allow_pickle=True)

# Load labels from labels.txt
with open('labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize cvzone HandDetector
hand_detector = HandDetector(detectionCon=0.5, maxHands=2)

# Global variables
frames = []
recognition_active = False
recognized_text = ""
translated_text = ""
language = "en"
sequence_length = 30  # Number of frames for gesture recognition
num_landmarks = 42  # 21 landmarks * 2 hands
sequence_features = []
prev_hand_landmarks = None  # To store previous hand landmarks for hand movement

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

def generate_frames():
    global frames, recognition_active, recognized_text, translated_text, sequence_features, prev_hand_landmarks
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if recognition_active:
                # Detect hands using cvzone HandDetector
                hands, img = hand_detector.findHands(frame, flipType=True)
                if hands:
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
                    
                    # Mirror the landmarks if only one hand is used
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
                    sequence_features.append(combined_features)
                    sequence_features = sequence_features[-sequence_length:]  # Keep the last 'sequence_length' elements
                    prev_hand_landmarks = np.array(hand_landmarks)  # Update previous landmarks
                    
                    if len(sequence_features) == sequence_length:
                        # Prepare the sequence for prediction
                        features_input = np.array(sequence_features)
                        features_input = features_input / 255.0  # Normalize features
                        features_input = np.expand_dims(features_input, axis=0)  # Add batch dimension

                        # Make prediction
                        prediction = cnn_lstm_model.predict(features_input)
                        predicted_label = labels[np.argmax(prediction)]
                        recognized_text += f" {predicted_label}"
                        
                        # Translate text (dummy translation for demonstration)
                        translated_text = Translator().translate(recognized_text, dest=language).text
                        
                        # Display predictions on the frame
                        cv2.putText(frame, f"Recognized: {predicted_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        sequence_features=[]
            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_language', methods=['POST'])
def set_language():
    global language
    data = request.json
    language = data['language']
    return jsonify({"message": "Language set successfully."})

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global recognition_active
    recognition_active = True
    return jsonify({"message": "Recognition started."})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global recognition_active
    recognition_active = False
    return jsonify({"message": "Recognition stopped."})
    sequence_features = []

@app.route('/recognize', methods=['POST'])
def recognize():
    global recognized_text, translated_text
    return jsonify({"recognized": recognized_text, "translated": translated_text})

@app.route('/clear', methods=['POST'])
def clear():
    global recognized_text, translated_text
    recognized_text = ""
    translated_text = ""
    return jsonify({"message": "Text cleared."})

@app.route('/backspace', methods=['POST'])
def backspace():
    global recognized_text
    recognized_text = recognized_text.rstrip(' ').rsplit(' ', 1)[0]
    return jsonify({"recognized": recognized_text})

def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    app.run(debug=True)
