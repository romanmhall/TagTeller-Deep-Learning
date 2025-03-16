from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import application_models  # Import application_models.py

app = Flask(__name__)

# Define paths
TRAINING_DATA_FOLDER = 'training_uploads'
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload_training_images', methods=['POST'])
def upload_images():
    """Handles uploading images for training"""
    try:
        brand = request.form.get('brand')
        if not brand:
            return jsonify({'error': 'No brand name provided'}), 400

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        brand_dir = os.path.join(TRAINING_DATA_FOLDER, brand)
        os.makedirs(brand_dir, exist_ok=True)

        count = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(brand_dir, filename)
                file.save(file_path)
                count += 1

        return jsonify({
            'success': True,
            'count': count,
            'brand': brand,
            'upload_location': os.path.abspath(brand_dir)
        })
    except Exception as e:
        print(f"Error uploading images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_brand', methods=['POST'])
def detect_brand():
    """Handles brand detection using application_models.py"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Save the uploaded image temporarily
        temp_path = "temp_upload.jpg"
        file.save(temp_path)

        # Use application_models.py to classify the image
        predicted_brand, confidence = application_models.classify_uploaded_image(temp_path)

        # Delete temp image after processing
        os.remove(temp_path)

        if confidence < 50:  # Confidence threshold
            return jsonify({'brand': 'Unknown', 'confidence': 0, 'message': 'Confidence too low to make a prediction'})

        return jsonify({
            'brand': predicted_brand,
            'confidence': float(round(confidence, 2))  # ✅ Convert float32 → float
        })

    except Exception as e:
        print(f"Error detecting brand: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
