
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#Dataset
class MarineDebrisSTACDataset(Dataset):
    def __init__(self, stac_url, transform=None, api_key=None):
        self.stac_url = stac_url
        self.transform = transform
        self.api_key = api_key

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Query STAC API for items
        response = requests.get(f"{stac_url}/items", headers=headers)
        response.raise_for_status()
        self.items = response.json()["features"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Get image URL
        img_url = item["assets"]["image"]["href"]
        img_response = requests.get(img_url)
        img = Image.open(BytesIO(img_response.content)).convert("RGB")

        # Get bounding boxes + labels
        annotations = item["properties"]["annotations"]
        boxes = torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32)
        labels = torch.tensor([ann["label_id"] for ann in annotations], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target
#Model
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

#Training
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

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

    torch.save(model.state_dict(), "models/marine_debris_detector.pth")
    print("Model saved to models/marine_debris_detector.pth")

#Predictor
def predict_image(image_url, num_classes=6, api_key=None):
    model = build_model(num_classes)
    model.load_state_dict(torch.load("models/marine_debris_detector.pth"))
    model.eval()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(image_url, headers=headers)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img)

    print("Predicted boxes:", prediction[0]["boxes"])
    print("Predicted labels:", prediction[0]["labels"])
    return prediction

