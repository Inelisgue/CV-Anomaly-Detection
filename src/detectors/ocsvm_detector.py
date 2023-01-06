from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

class OCSVMDetector:
    def __init__(self, kernel="rbf", nu=0.1, gamma="auto"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.scaler = StandardScaler()

    def train(self, X_normal):
        # Scale data before training
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)

    def predict(self, X):
        # Scale new data using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def detect_anomalies(self, X, threshold=0):
        predictions = self.predict(X)
        # OneClassSVM returns -1 for outliers and 1 for inliers
        anomalies = np.where(predictions == -1)[0]
        return anomalies

    def get_decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
