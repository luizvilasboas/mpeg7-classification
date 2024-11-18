import json
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def segment_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, segmented = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return segmented


def extract_features(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / \
            (perimeter ** 2) if perimeter > 0 else 0
        features.append([area, perimeter, circularity])
    return np.mean(features, axis=0) if features else [0, 0, 0]


def load_dataset(data_path):
    labels = []
    features = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            segmented_image = segment_image(file_path)
            feature_vector = extract_features(segmented_image)
            features.append(feature_vector)
            labels.append(label)
    return np.array(features), np.array(labels)


def create_metrics_folder():
    if not os.path.exists("metrics"):
        os.makedirs("metrics")


def save_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"metrics/{model_name}_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(
        f"Relatório de classificação salvo: metrics/{model_name}_classification_report.json")


def save_confusion_matrix(y_true, y_pred, model_name, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(f"metrics/{model_name}_confusion_matrix.png")
    plt.close()
    print(
        f"Matriz de confusão salva: metrics/{model_name}_confusion_matrix.png")


def plot_feature_graphs(features, labels, feature_names):
    create_metrics_folder()
    data = pd.DataFrame(features, columns=feature_names)
    data["Class"] = labels

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=feature_name, y=feature_names[(
            i+1) % 3], hue="Class", palette="viridis", s=100)
        plt.title(f"Gráfico de {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel(feature_names[(i+1) % 3])
        plt.tight_layout()
        output_path = f"metrics/{feature_name}_scatter_plot.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Gráfico {feature_name} salvo: {output_path}")


def plot_features_pairgrid(features, labels, feature_names):
    create_metrics_folder()
    data = pd.DataFrame(features, columns=feature_names)
    data["Class"] = labels
    pair_grid = sns.PairGrid(
        data, hue="Class", palette="viridis", diag_sharey=False)
    pair_grid.map_upper(sns.scatterplot)
    pair_grid.map_lower(sns.kdeplot, fill=True)
    pair_grid.map_diag(sns.histplot, kde=True)
    pair_grid.add_legend()
    plt.tight_layout()
    output_path = "metrics/features_pairgrid.png"
    pair_grid.savefig(output_path)
    plt.close()
    print(f"Gráfico PairGrid salvo: {output_path}")


if __name__ == "__main__":
    dataset_path = "dataset"

    features, labels = load_dataset(dataset_path)

    feature_names = ["Área", "Perímetro", "Circularidade"]

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    plot_feature_graphs(features_normalized, labels, feature_names)

    plot_features_pairgrid(features_normalized, labels, feature_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    rf = RandomForestClassifier(random_state=42)

    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    create_metrics_folder()

    for model, name in [(knn, "k-NN"), (rf, "Random Forest")]:
        y_pred = model.predict(X_test)
        save_classification_report(y_test, y_pred, name)
        save_confusion_matrix(y_test, y_pred, name, classes=np.unique(labels))
