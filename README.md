# 🧠 Gender-Based Participant Counting from Images

## 📌 Problem Statement

This project builds a **Machine Learning model** to:

* Detect faces in an image
* Classify each face as:

  * `0 → Male`
  * `1 → Female`
* Count total number of male and female participants

---

## 🚀 Features

* ✅ Face Detection + Gender Classification
* ✅ CPU-only execution (as required)
* ✅ PyTorch `.pth` model format
* ✅ No internet required during inference
* ✅ Lightweight & fast

---

## 🏗️ Project Structure

```
gender-counter/
│
├── data/
│   ├── train/
│   ├── val/
│
├── models/
│   └── gender_model.pth
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── dataset.py
│   └── model.py
│
├── utils/
│   └── face_detect.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 📊 Dataset

Use this dataset:

👉 https://www.kaggle.com/datasets/jangedoo/utkface-new

### Dataset Info:

* Contains face images labeled with age, gender, ethnicity
* Gender:

  * `0 → Male`
  * `1 → Female`

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/vaishnavijagtap02/Gender_Classification.git
cd Gender_Classification
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```
torch
torchvision
opencv-python
numpy
Pillow
```

---

## 🧠 Model Architecture

* Pretrained: `ResNet18`
* Modified final layer → Binary classification
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

## 🏋️ Training

```bash
python src/train.py
```

Output:

* `gender_model.pth`

---

## 🔍 Inference (Prediction)

```bash
python main.py --image sample.jpg
```

### Output Example:

```
Male: 3
Female: 2
```

---

## ⚡ Evaluation Criteria (Followed)

* Accuracy ✔
* F1 Score ✔
* Fast inference ✔
* Small model ✔
* Noise robustness ✔

---

## 🧾 Model Card

### Dataset

* UTKFace Dataset

### Architecture

* ResNet18 (Modified)

### Parameters

* ~11 Million

### Training Strategy

* Transfer Learning
* Data Augmentation (Flip, Resize)

### Ethical Considerations

* Gender classification may introduce bias
* Dataset imbalance handled with augmentation

---

## ⚠️ Constraints Followed

* ✅ PyTorch `.pth` only
* ✅ CPU compatible
* ❌ No TensorFlow / ONNX
* ❌ No internet during execution

---

## 💡 Future Improvements

* Multi-face detection improvements
* Real-time webcam support
* Better bias mitigation

---

## 🙌 Author

Vaishnavi Jagtap
