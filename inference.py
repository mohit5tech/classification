import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Define constants
IMG_SIZE = (224, 224)  # Match the size used during training
CLASS_NAMES = ['Pleural', 'Cardiomegaly']  # Match the class names from training

# Load the trained model
MODEL_PATH = 'path/to/saved_model.h5'  # Replace with your saved model path
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)  # Resize to match model input
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize (if required during training)
    return img_array

# Function to make predictions on an image
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

# Example usage
if __name__ == '__main__':
    # Path to the directory containing test images
    TEST_IMAGE_DIR = 'path/to/test_images/'  # Replace with your test image directory

    # Iterate over test images and make predictions
    for image_name in os.listdir(TEST_IMAGE_DIR):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)
        predicted_class, confidence = predict_image(image_path)
        print(f"Image: {image_name}, Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
