import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from manager_models import ModelManager

# Define labels folder path
LABELS_FOLDER = r"C:\Users\roman\PycharmProjects\pythonProject3\labels"

# Ensure labels directory exists
if not os.path.exists(LABELS_FOLDER):
    print(f"WARNING: Labels directory {LABELS_FOLDER} not found. Creating it now...")
    os.makedirs(LABELS_FOLDER)

# Get class names and strip "_model_class" suffix
class_names = sorted([
    folder.replace("_model_class", "")  # Remove _model_class suffix
    for folder in os.listdir(LABELS_FOLDER)
    if os.path.isdir(os.path.join(LABELS_FOLDER, folder))
])

if not class_names:
    print("WARNING: No valid labels found in the 'labels/' directory.")

# Load the best trained model
model_path = ModelManager.get_best_model()

if model_path and os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model successfully loaded.\n")
else:
    raise FileNotFoundError(f"ERROR: Model not found at {model_path}. Please train the model first.")


def classify_uploaded_image(img_path):
    """
    Classifies a single uploaded image using the trained model.

    :param img_path: Path to the image file.
    :return: Predicted brand and confidence score.
    """
    if model is None:
        return "No Model Loaded", 0

    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Make a prediction
        predictions = model.predict(img_array)
        top_index = np.argmax(predictions[0])
        confidence = predictions[0][top_index] * 100

        # Ensure valid index range
        if not class_names or top_index >= len(class_names):
            return "Unknown", 0

        predicted_brand = class_names[top_index]  # Now correctly mapped without "_model_class"
        return predicted_brand, confidence

    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        return "Error", 0
