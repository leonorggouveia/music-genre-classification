# Music Genre Classification Project - Main Script and Traditional ML Pipeline

# Imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
import joblib

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


# Configuration and Parameters
root_dir = "GTZAN"         # Folder for the dataset
sr = 22050                 # Sampling rate
duration = 30              # Clip length in seconds
n_fft = 2048               # FFT window size
hop_length = 512           # Step size between FFT frames
random_state = 42          # Seed for reproducibility
k_clusters = 10            # Number of clusters (equal to number of genres)
results_dir = "results"    # Output directory
features_csv = os.path.join(results_dir, "features_gtzan.csv")


# Exploratory Data Analysis and Clustering
def clustering_and_visualization(df):
    """Applies Standard Scaling, PCA, and K-means clustering to the feature space"""
    X = df.drop("genre", axis=1).values # Feature matrix
    y = df["genre"].values              # Target labels

    # Data Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Unsupervised Clustering (K-Means)
    kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Clustering Evaluation Metrics
    sil = silhouette_score(X_scaled, clusters) # Measures cohesion and separation
    ari = adjusted_rand_score(pd.factorize(y)[0], clusters) # Measures similarity to ground truth
    nmi = normalized_mutual_info_score(pd.factorize(y)[0], clusters) # Measures shared information with ground truth
    print(f"Clustering metrics: Silhouette={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # PCA Scatter Plot Colored by True Genre
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=palette, s=40, alpha=0.8)
    plt.title("PCA projection colored by true genre")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pca_projection_true_genre.png"))
    plt.show()

    # PCA Scatter Plot Colored by K-Means Cluster Assignment
    plt.figure(figsize=(10, 8))
    cluster_labels = [f"Cluster {c}" for c in clusters]
    palette = sns.color_palette("tab10", n_colors=k_clusters)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette=palette, s=40, alpha=0.8)
    plt.title("PCA projection colored by K-Means Cluster")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pca_projection_kmeans_cluster.png"))
    plt.show()

    return X_scaled, y


# Classification and Evaluation
def classification_and_evaluation(X_scaled, y):
    """Trains and evaluates Random Forest and SVM"""

    # Data split: Stratified 80% train and 20% test
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=random_state)

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    print("\nRandom Forest Report")
    print(classification_report(y_te, y_pred_rf))

    rf_model_path = os.path.join(results_dir, "rf_model.joblib")
    joblib.dump(rf, rf_model_path)
    print(f"Saved Random Forest model to {rf_model_path}")

    # Support Vector Machine (SVM) Classifier
    # Uses Radial Basis Function (RBF) kernel
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=random_state)
    svm.fit(X_tr, y_tr)
    y_pred_svm = svm.predict(X_te)
    y_proba_svm = svm.predict_proba(X_te) # Probabilities for ROC curve
    print("\nSVM Report")
    print(classification_report(y_te, y_pred_svm))

    svm_model_path = os.path.join(results_dir, "svm_model.joblib")
    joblib.dump(svm, svm_model_path)
    print(f"Saved SVM model to {svm_model_path}")

    # Calculate Class-Wise Accuracy for Comparison
    print("\nClass-wise Accuracy Improvements")
    genres = np.unique(y_te)
    # Calculate accuracy for each genre for both classifiers
    df_acc = pd.DataFrame({
        "Genre": genres,
        "RF": [accuracy_score(y_te[y_te == g], y_pred_rf[y_te == g]) for g in genres],
        "SVM": [accuracy_score(y_te[y_te == g], y_pred_svm[y_te == g]) for g in genres],
    })
    print(df_acc)
    # Identify which genres benefited most from the SVM
    print("\nGenres that improved:", df_acc[df_acc["SVM"] > df_acc["RF"]]["Genre"].tolist())

    # Plot Grouped Bar Chart for RF vs SVM Accuracy
    df_acc_melt = df_acc.melt(id_vars='Genre', var_name='Classifier', value_name='Accuracy')
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Genre', y='Accuracy', hue='Classifier', data=df_acc_melt, palette={'RF': 'green', 'SVM': 'blue'})
    plt.title("Class-wise Accuracy Comparison: Random Forest vs. SVM")
    plt.ylabel("Accuracy")
    plt.xlabel("Music Genre")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(title='Classifier')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_wise_accuracy_comparison.png"))
    plt.show()

    # Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix for Random Forest
    sns.heatmap(confusion_matrix(y_te, y_pred_rf, labels=rf.classes_), annot=True, fmt="d",
                xticklabels=rf.classes_, yticklabels=rf.classes_, cmap="Greens", ax=axes[0])
    axes[0].set_title("Random Forest Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Confusion Matrix for SVM
    sns.heatmap(confusion_matrix(y_te, y_pred_svm, labels=svm.classes_), annot=True, fmt="d",
                xticklabels=svm.classes_, yticklabels=svm.classes_, cmap="Blues", ax=axes[1])
    axes[1].set_title("SVM Confusion Matrix")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_comparison.png"))
    plt.show()

    # Plot One-vs-Rest ROC Curve for SVM
    lb = LabelBinarizer()
    y_te_bin = lb.fit_transform(y_te)
    classes = lb.classes_
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_te_bin[:, i], y_proba_svm[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(sns.color_palette("tab10", n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curve for SVM')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "svm_ovr_roc_curve.png"))
    plt.show()

    # K-fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(svm, X_scaled, y, cv=skf)
    print(f"\n5-Fold Cross-validation Accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")

    return svm, X_te, y_te


def explainability(model, X_te, y_te, feature_names, top_k=10):
    """Uses Permutation Importance on SVM to identify the most influential audio features"""

    # Calculate Permutation Importance
    res = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=random_state, n_jobs=-1)
    importances = res.importances_mean
    indices = np.argsort(importances)[::-1]
    top_idx = indices[:top_k]

    print("\nTop Features by Permutation Importance:")
    for i in top_idx:
        print(f"{feature_names[i]}: {importances[i]:.5f}")

    # Plot of the Top Features
    plt.figure(figsize=(10, 6))
    feature_labels = [feature_names[i] for i in top_idx]
    sns.barplot(x=importances[top_idx], y=feature_labels, hue=feature_labels, palette="viridis",
                dodge=False, legend=False)
    plt.title("Top Feature Importances (Permutation Importance)")
    plt.xlabel("Mean Decrease in Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importances.png"))
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Import required functions from the features and CNN modules
    from utils_features import build_feature_csv, get_feature_names
    from deep_cnn_module import train_cnn_on_logmels, plot_cnn_history

    # Setup Environment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Data Loading and Feature Extraction
    df = build_feature_csv(root_dir, features_csv, overwrite=False)
    print("Dataset counts per genre:\n", df["genre"].value_counts())

    # Exploratory Analysis and Preprocessing
    X_scaled, y = clustering_and_visualization(df)

    # Traditional Machine Learning Pipeline
    svm, X_te, y_te = classification_and_evaluation(X_scaled, y)

    # Explainability
    feature_names = get_feature_names()
    explainability(svm, X_te, y_te, feature_names, top_k=12)

    # Deep Learning Pipeline Execution
    model, le, history = train_cnn_on_logmels(root_dir)

    # Deep Learning Training History Plot
    plot_cnn_history(history, results_dir)
