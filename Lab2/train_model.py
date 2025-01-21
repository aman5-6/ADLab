import os
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
def load_data(data_dir):
    X, y = [], []
    for label, class_dir in enumerate(['cats', 'dogs']):
        class_path = os.path.join(data_dir, class_dir)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (64, 64))  # Resize images
            X.append(img.flatten())  # Flatten image
            y.append(label)  # 0 for cat, 1 for dog
    return np.array(X), np.array(y)

# Train SVM
def train_svm(X, y):
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    with open('backend/models/svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train Random Forest
def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    with open('backend/models/random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train Logistic Regression
def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    with open('backend/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train K-Means
def train_kmeans(X):
    model = KMeans(n_clusters=2)
    model.fit(X)
    with open('backend/models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train CNN
def train_cnn(X, y):
    X = X.reshape(-1, 64, 64, 3) / 255.0  # Normalize and reshape
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save('backend/models/cnn_model.h5')

if __name__ == '__main__':
    data_dir = 'data/train'
    X, y = load_data(data_dir)
    train_svm(X, y)
    train_random_forest(X, y)
    train_logistic_regression(X, y)
    train_kmeans(X)
    train_cnn(X, y)
    print("All models trained and saved.")
