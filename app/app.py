# app/app.py

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import sys
import os
import json
import numpy as np
import cv2
import io
from PIL import Image as PILImage

# Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.grad_cam import GradCAM

# Paths
MODEL_PATH = "./models/resnet18_neu.pth"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
ACCURACY_PATH = "models/accuracy.txt"
CONF_MATRIX_PATH = "models/confusion_matrix.png"
CLASS_DIST_PATH = "models/class_distribution.json"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model once using Streamlit caching and measure load time
@st.cache_resource
def load_model_with_timer():
    start_time = time.time()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    end_time = time.time()
    load_time = end_time - start_time
    return model, load_time

# Load model
model, model_load_time = load_model_with_timer()

# Set up Streamlit page
st.set_page_config(page_title="Surface Defect Detector", layout="centered")
st.title("🧠 Surface Defect Detection with Grad-CAM")
st.write("Upload a steel surface image to detect and visualize possible defects.")

# Show model loading time
st.info(f"⏱️ Model loaded in **{model_load_time:.2f} seconds**")

# =========================
# 🔍 Upload + Prediction UI
# =========================
uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼 Uploaded Image', use_container_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    # Display prediction
    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"🔎 Confidence: **{confidence * 100:.2f}%**")

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer=model.layer4)
    cam = gradcam.generate(img_tensor, class_idx=predicted.item())

    # Prepare visualization
    image_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 🔥 Heatmap Intensity Slider
    st.subheader("🔥 Adjust Heatmap Intensity")
    alpha = st.slider("Heatmap Intensity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    # Show Grad-CAM visualization
    st.image(overlay, caption="🔥 Grad-CAM Heatmap", use_container_width=True)

    # Download button
    overlay_pil = PILImage.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button("📥 Download Grad-CAM Heatmap", data=buf, file_name="gradcam_heatmap.jpg", mime="image/jpeg")

# =========================
# 📊 Sidebar Insights
# =========================
with st.sidebar:
    st.header("📊 Model Insights")
    st.markdown("---")

    # Validation Accuracy
    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH, "r") as f:
            acc = f.read().strip()
        st.metric(label="✅ Validation Accuracy", value=f"{acc}%")
    else:
        st.warning("⚠️ Accuracy not found. Please train the model.")

    # Class Distribution
    if os.path.exists(CLASS_DIST_PATH):
        with open(CLASS_DIST_PATH, "r") as f:
            dist_data = json.load(f)
        class_labels = list(dist_data.keys())
        class_counts = list(dist_data.values())
        st.subheader("📦 Class Distribution")
        st.bar_chart(data=dict(zip(class_labels, class_counts)))
    else:
        st.warning("⚠️ Class distribution not available.")

    # Confusion Matrix
    if os.path.exists(CONF_MATRIX_PATH):
        st.subheader("📉 Confusion Matrix")
        st.image(CONF_MATRIX_PATH, caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix not found. Please train the model.")








# =========================
# 🙋 About the Developer
# =========================
st.markdown(
    """
    <hr style='border: 1px solid #444;'>
<br>
<br>
<br>
<br>
<br>
    <div style='text-align: left; font-size: 15px; padding: 10px;'>
        <strong><pre>👨‍💻 Developed by <span style='color:#1f77b4;'>Souvik Dan</span></strong> 
        📧 <a href='mailto:souvikdan925@gmail.com'>souvikdan925@gmail.com</a>
        🔗 <a href='https://www.linkedin.com/in/souvik-dan' target='_blank'>LinkedIn Profile</a>
    </div></pre>
    """,
    unsafe_allow_html=True
)
