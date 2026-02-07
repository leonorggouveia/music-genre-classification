# Music Genre Classification Project - Deep Learning (CNN) Pipeline

# Imports
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras import layers, models


def extract_logmel(file_path):
    """Loads audio, generates a Log-Mel Spectrogram, ensures fixed size, and normalizes it"""

    # Load and resample audio signal to the target sample rate
    y, sr_loaded = librosa.load(file_path, sr=22050, duration=30)
    # Compute Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128,
                                        n_fft=2048, hop_length=512)
    # Convert to logarithmic Decibel (dB) scale
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Standardization
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)

    # Calculate the required width (number of frames)
    fixed_frames = int(np.ceil(30 * 22050 / 512))

    # Padding or cropping to ensure fixed temporal size
    if logmel.shape[1] < fixed_frames:
        pad = fixed_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode='constant')
    else:
        logmel = logmel[:, :fixed_frames]
    return logmel


def build_cnn(input_shape, num_classes):
    """Defines the Convolutional Neural Network architecture"""
    model = models.Sequential([
        # Feature extraction
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Deeper feature extraction
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Highest level features
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Classification Head
        layers.GlobalAveragePooling2D(), # Reduces spatial dimensions
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4), # Regularization
        layers.Dense(num_classes, activation='softmax') # Output layer for 10 genres
    ])

    # Configure optimization and loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model


def plot_cnn_history(history, results_dir):
    """Plots the training and validation loss and accuracy over epochs"""
    plt.figure(figsize=(12, 5))

    # Loss Evolution Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Evolution Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cnn_training_history.png"))
    plt.show()


def train_cnn_on_logmels(root_dir):
    """Main function to prepare data, train the CNN, and report test metrics"""

    # Identify genre folders for classification labels
    genres = sorted([g for g in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, g))])
    X, y = [], []

    # Data Preparation (Feature Generation)
    print("Extracting log-mel spectrograms")
    for genre in genres:
        for f in os.listdir(os.path.join(root_dir, genre)):
            if f.endswith('.wav'):
                path = os.path.join(root_dir, genre, f)
                try:
                    # Append the 2D Spectrogram data and the string label
                    X.append(extract_logmel(path))
                    y.append(genre)
                except Exception as e:
                    print("Error:", path, e)

    X = np.array(X)
    y = np.array(y)

    print("Dataset shape:", X.shape)
    X = X[..., np.newaxis] # Add channel dimension (required for Conv2D input)

    # Label Encoding and Splitting
    le = LabelEncoder()
    y_enc = le.fit_transform(y) # Convert string genres to integers

    # Stratified split: 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2,
                                                        stratify=y_enc, random_state=42)

    # Model Training Setup
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(genres))

    # Define callbacks
    # Halts training if validation performance stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    # Adjusts learning rate downward if loss plateaus
    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

    # Train the model over 50 epochs
    history = model.fit(
        X_train, y_train,
        validation_split=0.2, # Use 20% of the training data for validation during epochs
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, lr_sched],
        verbose=1
    )

    # Final Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.3f}")

    # Generate predictions on the unseen test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:\n")
    # Output metrics for each genre
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the final model
    cnn_model_path = os.path.join("results", "cnn_gtzan_model.h5")
    model.save(cnn_model_path)
    print(f"Saved CNN model to {cnn_model_path}")

    return model, le, history