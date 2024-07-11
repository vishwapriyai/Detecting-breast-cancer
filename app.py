from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

app = Flask("breast_cancer_classifier")  # Rename the Flask app

# Load breast cancer dataset
breast_cancer_dataset = load_breast_cancer()
X_train, y_train = breast_cancer_dataset.data, breast_cancer_dataset.target

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train_std.shape[1],)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_std, y_train, epochs=10)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']

    # Preprocess input data
    features = np.array(features).reshape(1, -1)
    features_std = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_std)[0]

    return jsonify({'prediction': int(np.argmax(prediction))})

if __name__ == '__main__':
    app.run(debug=True)
