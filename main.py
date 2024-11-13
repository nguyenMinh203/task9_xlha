import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time

# Đường dẫn tới dữ liệu ảnh
data_path = 'C:\\Users\\Admin\\PycharmProjects\\task9_xlha\\input\\animal'


# Đọc và tiền xử lý ảnh
def load_images(data_path):
    images = []
    labels = []
    label_names = os.listdir(data_path)
    for label, name in enumerate(label_names):
        folder_path = os.path.join(data_path, name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Chuyển kích thước ảnh về 64x64
            images.append(img.flatten())  # Chuyển thành vector 1D
            labels.append(label)
    return np.array(images), np.array(labels), label_names


# Tải dữ liệu
X, y, label_names = load_images(data_path)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hàm đánh giá mô hình với thời gian và độ chính xác
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    testing_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, training_time, testing_time


# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_accuracy, knn_train_time, knn_test_time = evaluate_model(knn_model, X_train, X_test, y_train, y_test)

# SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_accuracy, svm_train_time, svm_test_time = evaluate_model(svm_model, X_train, X_test, y_train, y_test)

# ANN
ann_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
ann_accuracy, ann_train_time, ann_test_time = evaluate_model(ann_model, X_train, X_test, y_train, y_test)

# In kết quả
print(f"KNN - Accuracy: {knn_accuracy:.2f}, Training Time: {knn_train_time:.2f}s, Testing Time: {knn_test_time:.2f}s")
print(f"SVM - Accuracy: {svm_accuracy:.2f}, Training Time: {svm_train_time:.2f}s, Testing Time: {svm_test_time:.2f}s")
print(f"ANN - Accuracy: {ann_accuracy:.2f}, Training Time: {ann_train_time:.2f}s, Testing Time: {ann_test_time:.2f}s")
