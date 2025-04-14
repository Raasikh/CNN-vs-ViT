# CNN-vs-ViT

# 🧠 Comparative Analysis of Vision Transformers and CNNs for Image Classification

## 📌 Overview
This project investigates the strengths and trade-offs between **Vision Transformers (ViTs)** and **Convolutional Neural Networks (CNNs)** for image classification.

We evaluate both models across:
- 🎯 **Accuracy**
- ⏱️ **Training Time**
- 💾 **Memory Usage**

---

## 🛠️ Tools & Frameworks
- **Language**: Python
- **Frameworks**: PyTorch, Torchvision, timm
- **Utilities**: scikit-learn, matplotlib
- **Environment**: Google Colab Pro (GPU)

---

## 🧪 Datasets Used

| Dataset              | Classes | Image Size | Complexity Level |
|----------------------|---------|------------|------------------|
| CIFAR-10             | 10      | 32x32      | Low              |
| CIFAR-100            | 100     | 32x32      | Medium           |
| Oxford-IIIT Pets     | 37      | Varies     | Medium           |
| Oxford Flowers-102   | 102     | Varies     | High             |

---

## 🧱 Model Architectures

### 🌀 CNN: ResNet-18
- Pretrained on ImageNet
- Final fully connected (FC) layer replaced
- All other layers frozen

### 🔭 ViT: Vision Transformer (Base, Patch Size 16)
- Pretrained on ImageNet
- Fine-tuned head layers
- Self-attention mechanism for global feature modeling

---

## ⚙️ Training Setup

| Model | Optimizer | Learning Rate | Loss Function      |
|-------|-----------|---------------|--------------------|
| CNN   | SGD       | 0.01          | CrossEntropyLoss   |
| ViT   | AdamW     | 0.0001        | CrossEntropyLoss   |

- **Epochs:** Up to 200 depending on dataset
- **Metrics Tracked:** Training/Validation/Test Accuracy, Training Time, Memory Usage

---

## 📊 Results Snapshot

### CIFAR-10
- **CNN:** ~78% accuracy  
- **ViT:** ⭐ ~95.5% accuracy

### CIFAR-100
- **CNN:** ~58% accuracy  
- **ViT:** ⭐ ~84% accuracy

### Oxford-IIIT Pets
- **CNN:** ~87% accuracy  
- **ViT:** ⭐ ~92% accuracy

### Oxford Flowers-102
- **CNN:** ~92% accuracy  
- **ViT:** ⭐ ~99% accuracy

> ⭐ ViTs outperform CNNs on all datasets in terms of accuracy and generalization.

---

## 🧠 Comparative Metrics

### 📈 Accuracy
ViTs achieved higher validation and test accuracies across all datasets.

### ⏱️ Training Time
CNNs were significantly faster per epoch (e.g., 56s vs 585s for CIFAR-100).

### 💾 Memory Usage
ViTs used more GPU memory due to attention mechanisms and deeper architecture.

---

## ⚖️ Trade-off Table

| Criteria         | CNN ✅                  | ViT ⭐                 |
|------------------|-------------------------|------------------------|
| Accuracy         | Good                    | **Superior**           |
| Training Speed   | **Faster**              | Slower                 |
| Memory Efficiency| **High**                | Lower                  |
| Generalization   | Decent                  | **Excellent**          |

---

## 📌 Conclusion
- ViTs provide **superior performance and generalization**, especially on complex datasets.
- CNNs remain **faster and more resource-efficient**, ideal for low-compute environments.
- For real-world applications requiring **precision**, ViTs are preferred.
- For scenarios with **compute constraints**, CNNs are a reliable choice.

---

## 🙏 Acknowledgements
Special thanks to **Prof. Lokesh Das** for his support.  
Inspired by the landmark paper:  
**"An Image is Worth 16x16 Words"** – [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## 🔗 References
- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [Oxford-IIIT Pets Dataset](https://paperswithcode.com/dataset/oxford-iiit-pets-1)  
- [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

---
