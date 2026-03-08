# 👩‍💻 Team InnovHERs (Team_40)

## Gender Classification Model

---

## 📌 Overview

This project implements a **Gender Classification Model** using the **MobileNetV3-Small** architecture.

The model is optimized for **CPU-only environments**, ensuring:

* ⚡ Fast inference speed
* 💻 Low computational requirements
* 🚀 Efficient real-time predictions

---

## 🏗️ Model Architecture

* **Architecture:** MobileNetV3-Small
* **Framework:** PyTorch
* **Model Format:** `.pth`
* **Input Image Size:** 224 × 224 pixels

---

## 🔄 Preprocessing Details

Before inference, images are:

* Resized to **224×224**
* Normalized using **ImageNet standards**

```
Mean: [0.485, 0.456, 0.406]  
Std:  [0.229, 0.224, 0.225]
```

---

## 🏷️ Label Mapping

| Label | Class  |
| ----- | ------ |
| 0     | Male   |
| 1     | Female |

---

## 🎯 Key Highlights

* Lightweight model
* Optimized for CPU deployment
* Fast and efficient performance
* Suitable for real-time applications

---
