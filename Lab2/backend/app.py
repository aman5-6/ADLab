from flask import Flask, request, render_template, jsonify
import os
import pickle
import cv2
import numpy as np
from keras.api.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'backend/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
MODELS = {
    'svm': pickle.load(open('backend/models/svm_model.pkl', 'rb')),
    'random_forest': pickle.load(open('backend/models/random_forest.pkl', 'rb')),
    'logistic_regression': pickle.load(open('backend/models/logistic_regression.pkl', 'rb')),
    'kmeans': pickle.load(open('backend/models/kmeans_model.pkl', 'rb')),
    'cnn': load_model('backend/models/cnn_model.h5')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    model_name = request.form['model']

    if file and model_name in MODELS:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (64, 64)).flatten() / 255.0
        img = np.array([img])

        model = MODELS[model_name]
        if model_name == 'cnn':
            img = img.reshape(-1, 64, 64, 3)
            prediction = model.predict(img)
            label = 'Cat' if prediction[0] < 0.5 else 'Dog'
        elif model_name == 'kmeans':
            cluster = model.predict(img)
            label = 'Cat' if cluster[0] == 0 else 'Dog'
        else:
            prediction = model.predict(img)
            label = 'Cat' if prediction[0] < 0.5 else 'Dog'

        return jsonify({'prediction': label})
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)