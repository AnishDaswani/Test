import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()  # ensure binary mask
        return image, mask

# Model
class PollutionDetector(nn.Module):
    def __init__(self):
        super(PollutionDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.outconv(x))
        return x

# Training
def train_model(image_dir="data/images", mask_dir="data/masks"):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = PollutionDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(5):
        for imgs, masks in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pollution_detector.pth")
    print("Model saved to models/pollution_detector.pth")

# Prediction
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    model = PollutionDetector()
    model.load_state_dict(torch.load("models/pollution_detector.pth"))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        prediction = model(image)
    print("Prediction mask shape:", prediction.shape)
    return prediction

# Main
if __name__ == "__main__":
    # Train on uploaded dataset
    train_model("data/images", "data/masks")

    # Run a test prediction
    predict_image("data/images/example.png")

