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
Gender_Classification/
│
├── dataset/              │
├── model/
│   └── model.pth
│
├── inference.py
├── predict.py
│
├── model_card.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **UTKFace Dataset**.

🔗 Download dataset:
[https://www.kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### 📁 Dataset Setup

After downloading, extract and place it as:

```
dataset/
 ├── Training/
 ├── Validation/
```

⚠️ Note:

* The dataset is **not included** in this repository due to size limitations.
* Please download it manually from the link above.

### 📌 Dataset Details

* Face images labeled with age, gender, ethnicity
* Gender labels:

  * `0 → Male`
  * `1 → Female`

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

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
python train.py
```

Output:

* `model.pth`

---

## 🔍 Inference (Prediction)

Make sure the model file exists at:

```
models/model.pth
```

Run:

```bash
python inference.py --image sample.jpg
```

### Output Example:

```
Male: 3
Female: 2
```

---

## 🧠 Model Loading

```python
import torch

model = torch.load("models/model.pth", map_location=torch.device('cpu'))
model.eval()
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
  n- Real-time webcam support
* Better bias mitigation

---

## 🙌 Author

**Vaishnavi Jagtap**
