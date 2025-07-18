# import torch
# import torchvision
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# print("✅ PyTorch version:", torch.__version__)
# print("✅ Torchvision version:", torchvision.__version__)
# print("✅ OpenCV version:", cv2.__version__)

# # Create a dummy image (black square with white border)
# img = np.zeros((100, 100, 3), dtype=np.uint8)
# img = cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)

# # Convert BGR (OpenCV) to RGB (Matplotlib)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Display the image using matplotlib
# plt.imshow(img_rgb)
# plt.title("Test Image")
# plt.axis("off")
# plt.show()




import torch
from torchvision import models

try:
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 classes

    model.load_state_dict(torch.load("models/resnet18_neu.pth", map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:")
    print(e)
