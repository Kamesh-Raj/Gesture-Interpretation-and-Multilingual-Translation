# Gesture Interpretation and Multilingual Translation

A real-time gesture recognition system that interprets sign language gestures and provides multilingual translation capabilities. This project uses deep learning (CNN-LSTM) to recognize sign language gestures and translates them into multiple languages.

## Features

- Real-time gesture recognition using webcam
- Support for 10 basic sign language gestures:
  - Bad
  - Correct
  - Good
  - Good Bye
  - Hello
  - Name
  - No
  - Please
  - Thanks
  - Yes
- Multilingual translation support
- Web-based interface for easy interaction
- CNN-LSTM model for accurate gesture recognition

## Technologies Used

- Python 3.x
- TensorFlow
- OpenCV
- MediaPipe
- Flask
- NumPy
- scikit-learn
- Google Translate API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kamesh-Raj/Gesture-Interpretation-and-Multilingual-Translation.git
cd Gesture-Interpretation-and-Multilingual-Translation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `app.py`: Main Flask application for the web interface
- `train.py`: Script for training the CNN-LSTM model
- `evaluate.py`: Script for evaluating model performance
- `preprocess.py`: Data preprocessing utilities
- `create_dataset.py`: Script for creating the training dataset
- `static/`: Contains CSS and JavaScript files for the web interface
- `templates/`: Contains HTML templates for the web interface

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Allow camera access when prompted

4. Perform gestures in front of your camera

5. The recognized gesture will be displayed along with its translation

## Model Training

To train the model on your own dataset:

1. Prepare your dataset following the format in `create_dataset.py`
2. Run the training script:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand tracking capabilities
- TensorFlow for deep learning framework
- Google Translate API for translation services 