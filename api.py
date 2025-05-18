from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

# Load the saved TensorFlow model
model = load_model("my_model.keras")  # Use the same model as main.py
label_list = ['No Tumor', ' Benign Tumor', 'Malignant Tumor']


def process_video(file):
    file_path = os.path.join('tmp', file.filename)
    file.save(file_path)

    frames_dir = os.path.join('tmp', 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    video = cv2.VideoCapture(file_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frame_count += 1
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    video.release()
    os.remove(file_path)

    return frames_dir, frame_count


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # or (299, 299) depending on model
    img = img / 255.0
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    return img



def create_video(frames_dir, frame_count, fps=5):
    frame_path = os.path.join(frames_dir, "frame_1.jpg")
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    output_path = os.path.join('tmp', 'output1.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(1, frame_count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    return output_path


def func(file):
    file_path = os.path.join('tmp', file.filename)
    file.save(file_path)
    # Preprocess as in main.py
    img = image.load_img(file_path, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 1)
    img_array /= 255.0
    probs = model.predict(img_array)
    print(f"Model output probabilities: {probs}")
    pl = np.argmax(probs, axis=-1)
    os.remove(file_path)
    return label_list[pl[0]]


@app.route('/ipredict', methods=['POST'])
def ipredict():
    if 'image' not in request.files:
        return jsonify({"status": "No file provided"}), 400

    image_file = request.files['image']
    if not image_file.filename.lower().endswith('.jpg'):
        return jsonify({"status": "Invalid file type, must be .jpg"}), 400

    prediction = func(image_file)
    print(prediction)
    return jsonify({"status": prediction})


def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 1)
    img_array /= 255.0
    probs = model.predict(img_array)
    pl = np.argmax(probs, axis=-1)
    return label_list[pl[0]]

@app.route('/vpredict', methods=['POST'])
def vpredict():
    if 'video' not in request.files:
        return jsonify({"status": "No file provided"}), 400

    video_file = request.files['video']
    if not video_file.filename.lower().endswith('.mp4'):
        return jsonify({"status": "Invalid file type, must be .mp4"}), 400

    # Save the uploaded video to a temporary path
    file_path = os.path.join('tmp', video_file.filename)
    video_file.save(file_path)

    # Use the same process_video as in video_predict.py (accepts file path)
    frames_dir = os.path.join('tmp', 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    video = cv2.VideoCapture(file_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frame_count += 1
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    video.release()
    os.remove(file_path)

    predictions = []
    class_counts = {label: 0 for label in label_list}
    frame_labels = {label: [] for label in label_list}

    for i in range(1, frame_count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        prediction = predict_image(frame_path)
        predictions.append(prediction)
        class_counts[prediction] += 1
        frame_labels[prediction].append(f"frame_{i}.jpg")

    # Aggregated result: the class with the most frames
    aggregated_result = max(class_counts, key=class_counts.get)

    print("Aggregated Result:", aggregated_result)
    print("Total frames:", frame_count)
    print("Class counts:", class_counts)
    print("Frame labels:", frame_labels)

    shutil.rmtree(frames_dir)

    return jsonify({
        "status": aggregated_result,
        "total_frames": frame_count,
        "class_counts": class_counts,
        "frame_labels": frame_labels
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

