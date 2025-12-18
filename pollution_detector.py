import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from io import BytesIO
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import rasterio
import numpy as np

# Class names for marine debris
CLASS_NAMES = {
    1: "plastic",
    2: "wood",
    3: "algae",
    4: "sargassum", #large brown algae
    5: "artificial_item"
}

# Get data without downloading
class MarineDebrisSTACDataset(Dataset):
    def __init__(self, stac_url, transform=None, api_key=None):
        self.stac_url = stac_url
        self.transform = transform
        self.api_key = api_key

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Get dataset items
        query = {
            "collections": ["sentinel-2-l2a"],
            "bbox": [-81.5, 25.0, -80.5, 26.0],  # Florida coast example
            "datetime": "2025-06-01/2025-06-30",
            "limit": 10
        }
        response = requests.post(f"{stac_url}/search", json=query, headers=headers)
        response.raise_for_status()
        self.items = response.json()["features"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # getimage from API (Sentinel-2 band B04 as example)
        img_url = item["assets"]["B04"]["href"]
        with rasterio.open(img_url) as src:
            img_array = src.read(1)  # single band
        img = Image.fromarray(img_array).convert("RGB")

        # Remove bounding boxes and labels from annotations
        # Earth Search does not provide debris annotations, so we use dummy targets
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # Change image as specified
        if self.transform:
            img = self.transform(img)

        return img, target

# Fast model with customizable layers
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training
def train_model(stac_url, api_key=None, num_classes=6):
    transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
    dataset = MarineDebrisSTACDataset(stac_url, transform=transform, api_key=api_key)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = build_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        for images, targets in loader:
            images = list(img for img in images)
            targets = list(tgt for tgt in targets)

            # how far off from accurate model was
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/marine_debris_detector.pth")
    print("Model saved to models/marine_debris_detector.pth")

# Prediction
def predict_and_visualize(image_url, num_classes=6, api_key=None, score_thresh=0.5):
    model = build_model(num_classes)
    model.load_state_dict(torch.load("models/marine_debris_detector.pth"))
    model.eval()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # get image
    with rasterio.open(image_url) as src:
        img_array = src.read(1)
    img = Image.fromarray(img_array).convert("RGB")
    transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    # predict
    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = prediction[0]["scores"]

    # Draw bounding boxes and labels on image
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thresh:
            x1, y1, x2, y2 = box.tolist()
            label_name = CLASS_NAMES.get(label.item(), f"Class {label.item()}")
            draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
            draw.text((x1, y1), f"{label_name} ({score:.2f})", fill="red")

    img.show()
    return prediction

if __name__ == "__main__":
    STAC_URL = "https://earth-search.aws.element84.com/v1"
    API_KEY = None

    # Train model
    train_model(STAC_URL, api_key=API_KEY, num_classes=6)

    # Predict sample image (replace with actual asset href from search results)
    sample_image_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a/2025/6/1/tiles/10/T/ES/2025-06-01/0/B04.tif"
    predict_and_visualize(sample_image_url, num_classes=6, api_key=API_KEY)



