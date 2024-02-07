import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from train import InitializeModel

# Load YOLO model
yolo_model = InitializeModel.load_model('models/weights/best.pt')

# Load MiDaS model for depth estimation
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas_transforms = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_and_estimate_depth(image_path):
    # Load image
    img = Image.open(image_path)
    img_cv = cv2.imread(image_path)

    # Object Detection
    results = yolo_model(img)
    results.show()  # Show detected objects

    # Depth Estimation
    input_tensor = midas_transforms(img).unsqueeze(0)
    with torch.no_grad():
        depth = midas_model(input_tensor)

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_normalized = depth / depth.max()  # Normalize for visualization
    plt.imshow(depth_normalized.numpy(), cmap='inferno')
    plt.show()


# Example usage
detect_and_estimate_depth('images/4.jpg')
