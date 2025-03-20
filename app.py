from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

# Limit GPU memory growth (Prevents excessive memory allocation)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Reduce TensorFlow verbosity (Suppress warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load the pre-trained model
model = load_model('saved_model/emotion_recognition_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        prediction = model.predict(face_roi)
        emotion = emotion_labels[np.argmax(prediction)]
        emotions.append(emotion)

    return emotions

def generate_frames():
    camera_index = -1
    for i in range(5):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            camera_index = i
            cap.release()
            break
    else:
        print("No available camera found!")
        return

    # Open camera
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print("Error: Unable to access the camera.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to capture frame.")
                break  # Stop loop if frame is not captured

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = face_roi / 255.0  # Normalize

                # Make a prediction
                prediction = model.predict(face_roi)
                print("Model raw prediction:", prediction)

                if prediction is not None and len(prediction) > 0:
                    max_index = np.argmax(prediction)
                    
                    # print(f"Max index: {max_index}, Emotion Labels Length: {len(emotion_labels)}")
                    # print(f"Prediction Array: {prediction}")  # <-- Debugging line

                    if max_index < len(emotion_labels):
                        emotion = emotion_labels[max_index]
                    else:
                        emotion = "Neutral"

                else:
                    emotion = "No Face Detected"

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        print("Stream closed by client.")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        camera.release()  # Ensure camera is released properly
        print("Camera released.")

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is None:
                return "Error: Unable to read the uploaded image.", 400

            emotions = detect_emotion(image)

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_' + file.filename)
            cv2.imwrite(output_path, image)

            return render_template('result.html', original_image='uploaded_' + file.filename, emotions=emotions)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
