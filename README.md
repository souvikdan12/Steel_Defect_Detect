# 🧠 Surface Defect Detection using ResNet18

A Streamlit web app that detects **surface defects** in metal images using a fine-tuned ResNet18 model on the **NEU surface defect dataset**. Upload an image and get instant predictions for six types of common surface anomalies.

---

## 🚀 Demo

![App Screenshot](./screenshots/demo_ui.png)  
> Upload steel surface images and get real-time predictions directly from your browser.

---

## 🔍 Defect Classes

The model can detect the following 6 classes:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

---

## 🧠 Model Architecture

- Pretrained **ResNet18** on ImageNet
- Final `fc` layer modified for 6 classes
- Trained with PyTorch on `ImageFolder` structure
- Achieved **99.44% validation accuracy**

---

## 🗂️ Project Structure

