import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ----------------------------
# Model: ResNet18 + Two Heads
# ----------------------------
class ResNet18TwoHead(nn.Module):
    def __init__(self, num1: int, num2: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head1 = nn.Linear(feat_dim, num1)
        self.head2 = nn.Linear(feat_dim, num2)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head1(feat), self.head2(feat)

# ----------------------------
# Classifier Wrapper
# ----------------------------
class TwoDimImageClassifier:
    def __init__(self, model_dir="."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.ckpt_path = os.path.join(model_dir, "component_defect_model.pt")
        self.classes1 = []
        self.classes2 = []
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        os.makedirs(model_dir, exist_ok=True)

    def _save_checkpoint(self):
        torch.save({
            "model_state": self.model.state_dict(),
            "classes1": self.classes1,
            "classes2": self.classes2
        }, self.ckpt_path)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.classes1 = checkpoint["classes1"]
        self.classes2 = checkpoint["classes2"]
        self.model = ResNet18TwoHead(len(self.classes1), len(self.classes2)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, image_path):
        if self.model is None:
            self._load_checkpoint()
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits1, logits2 = self.model(x)
            prob1 = torch.softmax(logits1, dim=1)[0].cpu().numpy()
            prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
        idx1, idx2 = prob1.argmax(), prob2.argmax()
        avg_conf = (prob1[idx1] + prob2[idx2]) / 2.0
        result = {
            "component": self.classes1[idx1],
            "defect": self.classes2[idx2],
            "confidence": {
                "component_probs": dict(zip(self.classes1, prob1.tolist())),
                "defect_probs": dict(zip(self.classes2, prob2.tolist())),
                "average_confidence": avg_conf
            }
        }
        print("Prediction:", result)
        return result

    def train(self, samples, epochs=10, batch_size=8):
        # Build dynamic maps
        comp_set = sorted({s["component"] for s in samples})
        defect_set = sorted({s["defect"] for s in samples})
        comp_map = {c: i for i, c in enumerate(comp_set)}
        defect_map = {d: i for i, d in enumerate(defect_set)}
        self.classes1 = comp_set
        self.classes2 = defect_set
        self.model = ResNet18TwoHead(len(comp_set), len(defect_set)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()

        class DynamicDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                s = self.samples[i]
                img = Image.open(s["path"]).convert("RGB")
                img = self.transform(img)
                return img, torch.tensor(comp_map[s["component"]], dtype=torch.long), torch.tensor(defect_map[s["defect"]], dtype=torch.long)

        ds = DynamicDataset(samples, self.transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0
            for x, y1, y2 in dl:
                x, y1, y2 = x.to(self.device), y1.to(self.device), y2.to(self.device)
                optimizer.zero_grad()
                logits1, logits2 = self.model(x)
                loss = criterion(logits1, y1) + criterion(logits2, y2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self._save_checkpoint()
        print("Model saved successfully!")
        with open("training_loss.csv", "w") as f:
            f.write("epoch,loss\n")
            for i, l in enumerate(history, 1):
                f.write(f"{i},{l}\n")
        print("Loss history saved to training_loss.csv")

# ----------------------------
# Loader
# ----------------------------
def load_labeled_samples(image_root):
    samples = []
    summary = {}
    for folder in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if "_" in folder:
            comp, defect = folder.rsplit("_", 1)
        elif "-" in folder:
            comp, defect = folder.rsplit("-", 1)
        else:
            continue
        comp, defect = comp.strip(), defect.strip()
        count = 0
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({"path": os.path.join(folder_path, img_file), "component": comp, "defect": defect})
                count += 1
        if count > 0:
            summary[f"{comp}_{defect}"] = count

    total_images = sum(summary.values())
    print("\nImage Summary by Component-Defect Pair:")
    for k, v in sorted(summary.items()):
        percent = (v / total_images) * 100 if total_images > 0 else 0
        print(f"{k}: {v} images ({percent:.2f}%)")
    print(f"\nTotal Images Loaded: {total_images}")

    if summary:
        plt.figure(figsize=(12, 6))
        plt.barh(list(summary.keys()), list(summary.values()), color='steelblue')
        plt.xlabel('Number of Images')
        plt.title('Image Distribution by Component-Defect Pair')
        plt.tight_layout()
        plt.savefig("image_distribution.png")
        plt.show()

    return samples

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()
clf = TwoDimImageClassifier()

@app.get("/")
def root():
    return {"message": "Component-Defect Classifier API is running. Use /predict or /health."}

@app.get("/health")
def health():
    return {"status": "UP", "model_version": "component-defect-v1"}

@app.post("/predict")
async def predict(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    result = clf.predict(temp_path)
    os.remove(temp_path)
    return JSONResponse(content=result)

if __name__ == "__main__":
    image_root = r"C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\images"
    print("Loading labeled samples from:", image_root)
    samples = load_labeled_samples(image_root)
    print(f"Found {len(samples)} labeled images for training.")
    if len(samples) > 0:
        clf.train(samples, epochs=10, batch_size=8)
        print("Training complete.")
    else:
        print("No labeled images found for training.")
    print("Starting prediction server at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)