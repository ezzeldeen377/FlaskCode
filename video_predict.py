import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil

# Load the model and labels
model = load_model("my_model.keras")
label_list = ['No Tumor', ' Benign Tumor', 'Malignant Tumor']

def process_video(video_path):
    frames_dir = os.path.join('tmp', 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize frame to 299x299 for consistency with your API (can adjust if needed)
        frame = cv2.resize(frame, (299, 299))
        frame_count += 1
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    video.release()
    return frames_dir, frame_count

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 1)
    img_array /= 255.0
    probs = model.predict(img_array)
    pl = np.argmax(probs, axis=-1)
    return label_list[pl[0]]

def main(video_path):
    frames_dir, frame_count = process_video(video_path)

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

    return {
        "status": aggregated_result,
        "total_frames": frame_count,
        "class_counts": class_counts,
        "frame_labels": frame_labels
    }

if __name__ == "__main__":

    video_path = "XRecorder_20250518_01.mp4"
    result = main(video_path)
    print(result)