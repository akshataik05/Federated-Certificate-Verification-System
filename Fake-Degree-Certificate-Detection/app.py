import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Model Paths
MODEL_DIR = r"C:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection"
M1_PATH = os.path.join(MODEL_DIR, "fake_certificate_model.h5")
M2_PATH = os.path.join(MODEL_DIR, "model2.h5")
GLOBAL_PATH = os.path.join(MODEL_DIR, "global_model.h5")

print("Initializing TensorFlow models... This may take a few seconds.")
model1 = tf.keras.models.load_model(M1_PATH)
model2 = tf.keras.models.load_model(M2_PATH)
global_model = tf.keras.models.load_model(GLOBAL_PATH)
print("✅ Models loaded successfully.")

def format_prediction(score):
    # Keras binary classification: 0 = fake, 1 = real
    is_real = float(score) >= 0.5
    confidence = float(score) if is_real else 1.0 - float(score)
    label = "Real" if is_real else "Fake"
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": float(score)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Preprocess Image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # Base array map to [0, 255]
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) # shape (1, 224, 224, 3)
        
        # Manually scaled to [0, 1] for Model 2 
        # (Since Model 1 and Global Model inherently have a Rescaling layer built into the architecture)
        img_scaled = img_array / 255.0 
        
        # Predictions
        score1 = model1.predict(img_array, verbose=0)[0][0]
        score2 = model2.predict(img_scaled, verbose=0)[0][0]
        score_g = global_model.predict(img_array, verbose=0)[0][0]
        
        return jsonify({
            "client1": format_prediction(score1),
            "client2": format_prediction(score2),
            "global": format_prediction(score_g)
        })
        
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
