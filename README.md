# CV-Anomaly-Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)

A Python project implementing various computer vision techniques for anomaly detection in image and video streams. Focuses on industrial inspection, surveillance, and medical imaging applications. Utilizes deep learning and traditional CV methods.

## ✨ Features

-   **Autoencoder-based Anomaly Detection**: Leverage deep learning for unsupervised anomaly detection.
-   **One-Class SVM for Outlier Detection**: Implement traditional machine learning techniques for robust anomaly identification.
-   **Feature Extraction with Pre-trained CNNs**: Utilize powerful pre-trained models for effective feature representation.
-   **Real-time Inference Capabilities**: Design for efficient and timely anomaly detection in live streams.

## 🚀 Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Inelisgue/CV-Anomaly-Detection.git
    cd CV-Anomaly-Detection
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the example anomaly detection script (using MNIST as a proxy for normal data):

```bash
python src/main.py
```

This will train an autoencoder and then a One-Class SVM on the MNIST training data, and then report detected anomalies in the test set.

## 📚 Project Structure

```
CV-Anomaly-Detection/
├── src/
│   ├── models/             # Deep learning models (e.g., autoencoder.py)
│   ├── detectors/          # Traditional anomaly detection algorithms (e.g., ocsvm_detector.py)
│   └── main.py             # Main script for training and detection
├── data/                   # Dataset storage (e.g., MNIST data will be downloaded here)
├── configs/                # Configuration files
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
