# Music Genre Classification: ML vs. Deep Learning

Leonor Gouveia

## Project Overview

Developed during an Erasmus+ exchange program at **Universit?? degli Studi di Milano** for the **Audio Pattern Recognition** course. This project compares traditional Machine Learning approaches with Deep Learning architectures for automatic music genre classification.

The goal was to classify 1000 audio tracks from the GTZAN dataset into 10 distinct genres. The project involved building two parallel pipelines:

1.  **Feature-based:** Extracting handcrafted audio features for **SVM** and **Random Forest**.

2.  **End-to-End:** Using **Convolutional Neural Networks (CNN)** trained on **Log-Mel Spectrograms**.

## Technical Features

-   **Deep Learning:** TensorFlow/Keras (CNN architecture).

-   **Audio Processing:** Librosa (MFCCs, Spectral Centroid, Mel-Spectrograms), PyWavelets.

-   **Machine Learning:** Scikit-learn (SVM, Random Forest, PCA, K-Means).

-   **Explainable AI (XAI):** Permutation Importance to identify which audio features most influence classification.

## Results

-   **CNN Performance:** Achieved the highest accuracy (around 70%+), proving the power of spatial feature learning in spectrograms.

-   **Top Features:** Through Permutation Importance, MFCCs and Spectral Entropy were identified as the most critical features for genre differentiation.

-   **Unsupervised Analysis:** Used **PCA** and **K-Means** to visualize how genres cluster in a reduced feature space.

## Project Structure

-   `main_pipeline.py`: Manages the ML workflow and evaluation.

-   `deep_cnn_module.py`: CNN architecture and training loop for deep learning.

-   `utils_features.py`: Custom-built functions for advanced audio feature extraction.

-   `Project_Report.pdf`: A technical paper (English) detailing the methodology and scientific results.
