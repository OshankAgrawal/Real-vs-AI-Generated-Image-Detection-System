# 🧠 Real vs AI-Generated Image Detection System

This project aims to detect whether a given image is **real** (photographed) or **AI-generated** using deep learning techniques. It is built **from scratch** for better interpretability and understanding of how such classification models work.

## 🚀 Project Overview

With the rise of generative models like DALL·E, MidJourney, and Stable Diffusion, distinguishing between real and AI-generated images has become a major challenge. This system tackles that problem using a **custom-built Convolutional Neural Network (CNN)**.

## ✅ Features

* 🛠 Built from scratch (no pretrained models used)
* 📊 Custom CNN architecture for explainability
* 📁 Dataset split into Real and AI-Generated images
* 📈 Visualization of training progress (accuracy/loss)
* 🧪 Evaluation using accuracy, precision, recall, F1-score
* 📷 Accepts image input and predicts its authenticity

## 🧰 Tech Stack

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* TensorFlow / Keras
* Scikit-learn

## 📂 Project Structure

```
Real-vs-AI-Image-Detection/
│
├── dataset/
│   ├── real/              # Real images
│   └── ai/                # AI-generated images
│
├── model/
│   └── cnn_model.py       # Custom CNN architecture
│
├── notebook/
│   └── training.ipynb     # Jupyter notebook for training & evaluation
│
├── utils/
│   └── preprocessing.py   # Image preprocessing functions
│
├── saved_model/
│   └── model.h5           # Trained model weights
│
├── predict.py             # Prediction script for a single image
├── requirements.txt       # List of required packages
└── README.md              # You're reading it now!
```

## 🏁 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/OshankAgrawal/Real-vs-AI-Generated-Image-Detection-System.git
cd Real-vs-AI-Image-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Run the training notebook or `cnn_model.py` script to train the model using the provided dataset.

### 4. Predict on a new image

```bash
python predict.py --image path_to_your_image.jpg
```

## 📊 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 64.93% |
| Precision | 65% |
| Recall    | 65% |
| F1-Score  | 65% |


## 🙋‍♂️ Author

**Oshank Agrawal**
B.Tech – Artificial Intelligence & Data Science
Samrat Ashok Technological Institute, Vidisha (M.P.)
🌐 [LinkedIn]() • 📧 [oshankagrawal@gmail.com](mailto:oshankagrawal@gmail.com)

## 📌 Note

This project is for educational and research purposes. Model performance may vary depending on dataset quality and size.

## ⭐ Star This Repo

If you found this project helpful or interesting, consider giving it a ⭐ on GitHub to support the work!
