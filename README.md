# component-defect-classifier-ia

Confirmation

Your model and API are working as intended!  
- You selected an image from the `Circular Connection_Drummy` folder.
- The API returned:  
  ```json
  {"prediction": "Circular Connection_Drummy_0.85"}
  ```
- This matches the expected label and format.

Component Defect Classifier API
This README covers setup, code structure, usage, and troubleshooting.

```markdown
 Component Defect Classifier API

A FastAPI-based image classification service for predicting component and defect types from infrastructure images using a ResNet18-based PyTorch model.

 Features

- Multi-class, multi-head image classification (component & defect)
- REST API for prediction via file upload
- Swagger UI for easy testing (`/docs`)
- Human-friendly output: `Component_Defect_ConfidenceScore`
- Checkpointed model loading (Quick Mode)

 Directory Structure

```
ImageAnnot/
├── component_defect_classifier_v2.py    Main FastAPI app & model code
├── test_predict.py                      Example client script
├── component_defect_model.pt            Trained model checkpoint
├── images/                              Image folders (see below)
│   ├── Circular Connection_Drummy/
│   ├── Centre Support_Corrosion-Heavy/
│   └── ... (other component_defect folders)
└── ...
```

 1. Setup

 Python Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install fastapi uvicorn torch torchvision pillow matplotlib
  ```

 Directory Structure
- Place your trained model checkpoint as `component_defect_model.pt` in the same directory as `component_defect_classifier_v2.py`.
- Organize your images in folders named as `{Component}_{Defect}` (e.g., `Circular Connection_Drummy`).

 2. How It Works

 Model
- Uses a ResNet18 backbone with two classification heads:
  - Head 1: Predicts component type (e.g., Circular Connection, Channel Bracket, etc.)
  - Head 2: Predicts defect type (e.g., Drummy, Corrosion-Heavy, etc.)

 API Endpoints
- `GET /` — Welcome message
- `GET /health` — Health check
- `POST /predict` — Upload an image and get prediction

 3. Running the API Server

```bash
python component_defect_classifier_v2.py
```
- The server will auto-detect the model checkpoint and skip retraining if present.
- By default, it runs at `http://localhost:8000`.

 4. Making Predictions

 A. Using Swagger UI
- Open http://127.0.0.1:8000/docs in your browser.
- Use the `/predict` endpoint to upload an image.
- Response Example:
  ```json
  {
    "prediction": "Circular Connection_Drummy_0.85"
  }
  ```

 B. Using Python Client
```python
import requests
url = "http://127.0.0.1:8000/predict"
file_path = r"C:\path\to\your\image.jpg"
with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})
print(response.json()["prediction"])
```

 5. Code Overview

 component_defect_classifier_v2.py
- Model Definition: `ResNet18TwoHead` (PyTorch)
- Classifier Wrapper: Handles checkpoint loading, prediction, and training
- API: FastAPI app with `/predict` endpoint
- Prediction Output:  
  - Returns a string: `{component}_{defect}_{confidence score}` (e.g., `Circular Connection_Drummy_0.85`)

 test_predict.py
- Example script to test the API with a local image file.

 6. Troubleshooting

- KeyError: 'prediction'  
  Ensure you have restarted the FastAPI server after editing the `/predict` endpoint to return `{"prediction": formatted}`.
- Model Not Loading:  
  Make sure `component_defect_model.pt` exists in the working directory.
- API Returns 500:  
  Check FastAPI logs for errors (e.g., missing checkpoint, invalid image file).
- Wrong Labels:  
  Check that your test image is from the correct `{Component}_{Defect}` folder.

 7. Extending

- To return both the formatted string and full details, modify `/predict`:
  ```python
  return {
      "prediction": formatted,
      "raw": result
  }
  ```

 8. License

MIT License (or your preferred license)

 9. Contact

For questions, contact: [Your Name/Email]

You can copy-paste this README.md into your repo.  
If you want a more detailed section (e.g., for training, or for Docker), let me know!

Summary:  
- Your model and API are working and returning correct, human-friendly predictions.
- The README above documents the full workflow, code, and usage for your project.

Commented code
The comments explain:
Model architecture: Dual-head design for multi-task learning
Class methods: Purpose and parameters of key functions
Training pipeline: Data loading, preprocessing, and optimization
API endpoints: Request/response formats
Key decisions: Why frozen backbone, normalization values, etc.

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
    """
    Dual-head ResNet18 model for multi-task classification.
    Uses a shared ResNet18 backbone with two separate classification heads:
    - Head 1: Component classification
    - Head 2: Defect classification
    """
    def __init__(self, num1: int, num2: int):
        """
        Initialize the dual-head model.
        
        Args:
            num1: Number of component classes
            num2: Number of defect classes
        """
        super().__init__()
        # Load pre-trained ResNet18 weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Extract feature dimension from original final layer
        feat_dim = backbone.fc.in_features
        # Replace the original classification layer with identity (no-op)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Classification head 1: Component classifier
        self.head1 = nn.Linear(feat_dim, num1)
        # Classification head 2: Defect classifier
        self.head2 = nn.Linear(feat_dim, num2)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Tuple of (component_logits, defect_logits)
        """
        # Extract features using backbone
        feat = self.backbone(x)
        # Generate predictions from both heads
        return self.head1(feat), self.head2(feat)

# ----------------------------
# Classifier Wrapper
# ----------------------------
class TwoDimImageClassifier:
    """
    High-level wrapper for the ResNet18TwoHead model.
    Handles training, prediction, checkpointing, and preprocessing.
    """
    def __init__(self, model_dir="."):
        """
        Initialize the classifier.
        
        Args:
            model_dir: Directory for saving/loading model checkpoints
        """
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        # Path to model checkpoint file
        self.ckpt_path = os.path.join(model_dir, "component_defect_model.pt")
        # Lists to store unique class labels
        self.classes1 = []  # Component classes
        self.classes2 = []  # Defect classes
        # Model instance (lazy-loaded)
        self.model = None
        # Image preprocessing pipeline: resize, center crop, normalize
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # ImageNet normalization values
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _save_checkpoint(self):
        """Save model state, class labels, and metadata to disk."""
        torch.save({
            "model_state": self.model.state_dict(),
            "classes1": self.classes1,
            "classes2": self.classes2
        }, self.ckpt_path)

    def _load_checkpoint(self):
        """Load model state and class labels from checkpoint file."""
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        # Restore class labels
        self.classes1 = checkpoint["classes1"]
        self.classes2 = checkpoint["classes2"]
        # Recreate model architecture with correct dimensions
        self.model = ResNet18TwoHead(len(self.classes1), len(self.classes2)).to(self.device)
        # Load pre-trained weights
        self.model.load_state_dict(checkpoint["model_state"])
        # Set to evaluation mode (disable dropout, batch norm updates)
        self.model.eval()
        print("Model loaded from checkpoint for Quick Mode.")

    def predict(self, image_path):
        """
        Classify an image into component and defect categories.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Lazy-load model if not already loaded
        if self.model is None:
            self._load_checkpoint()
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        # Generate predictions without computing gradients
        with torch.no_grad():
            logits1, logits2 = self.model(x)
            # Convert logits to probabilities using softmax
            prob1 = torch.softmax(logits1, dim=1)[0].cpu().numpy()
            prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
        
        # Get predicted class indices (highest probability)
        idx1, idx2 = prob1.argmax(), prob2.argmax()
        # Calculate average confidence across both heads
        avg_conf = float((prob1[idx1] + prob2[idx2]) / 2.0)
        
        # Format results with all probabilities
        result = {
            "component": self.classes1[idx1],
            "defect": self.classes2[idx2],
            "confidence": {
                # Probability distribution for all components
                "component_probs": {k: float(v) for k, v in zip(self.classes1, prob1.tolist())},
                # Probability distribution for all defects
                "defect_probs": {k: float(v) for k, v in zip(self.classes2, prob2.tolist())},
                # Average confidence score
                "average_confidence": float(avg_conf)
            }
        }
        print("Prediction:", result)
        return result

    def train(self, samples, epochs=10, batch_size=8, freeze_backbone=True):
        """
        Train the model on labeled samples.
        
        Args:
            samples: List of dicts with keys 'path', 'component', 'defect'
            epochs: Number of training epochs
            batch_size: Batch size for training
            freeze_backbone: If True, only train classification heads
        """
        # Extract unique class labels and create mappings
        comp_set = sorted({s["component"] for s in samples})
        defect_set = sorted({s["defect"] for s in samples})
        comp_map = {c: i for i, c in enumerate(comp_set)}
        defect_map = {d: i for i, d in enumerate(defect_set)}
        # Store class labels for later use
        self.classes1 = comp_set
        self.classes2 = defect_set
        
        # Initialize model with correct number of output classes
        self.model = ResNet18TwoHead(len(comp_set), len(defect_set)).to(self.device)

        # Optionally freeze backbone to reduce training time and memory
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Training only classification heads.")

        # Optimizer: Adam with learning rate 5e-4
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=5e-4)
        # Loss function: Cross-entropy for multi-class classification
        criterion = nn.CrossEntropyLoss()

        # Custom dataset class for dynamic label mapping
        class DynamicDataset(torch.utils.data.Dataset):
            """Dataset that applies preprocessing and label encoding on-the-fly."""
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                """Get preprocessed image and encoded labels."""
                s = self.samples[i]
                img = Image.open(s["path"]).convert("RGB")
                img = self.transform(img)
                # Convert class names to indices
                return img, torch.tensor(comp_map[s["component"]], dtype=torch.long), torch.tensor(defect_map[s["defect"]], dtype=torch.long)

        # Create dataset and dataloader
        ds = DynamicDataset(samples, self.transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0
            # Iterate over batches
            for x, y1, y2 in dl:
                x, y1, y2 = x.to(self.device), y1.to(self.device), y2.to(self.device)
                # Reset gradients
                optimizer.zero_grad()
                # Forward pass
                logits1, logits2 = self.model(x)
                # Combine losses from both heads
                loss = criterion(logits1, y1) + criterion(logits2, y2)
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
                total_loss += loss.item()
            
            # Calculate average loss for epoch
            avg_loss = total_loss / len(dl)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save trained model
        self._save_checkpoint()
        print("Model saved successfully!")
        
        # Save training history to CSV for analysis
        with open("training_loss.csv", "w") as f:
            f.write("epoch,loss\n")
            for i, l in enumerate(history, 1):
                f.write(f"{i},{l}\n")
        print("Loss history saved to training_loss.csv")

# ----------------------------
# Loader
# ----------------------------
def load_labeled_samples(image_root):
    """
    Load images organized in folders with naming convention: 'Component_Defect'.
    
    Args:
        image_root: Root directory containing component-defect folders
        
    Returns:
        List of dicts with keys 'path', 'component', 'defect'
    """
    samples = []
    summary = {}
    
    # Iterate through all folders in image_root
    for folder in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder name to extract component and defect labels
        if "_" in folder:
            comp, defect = folder.rsplit("_", 1)
        elif "-" in folder:
            comp, defect = folder.rsplit("-", 1)
        else:
            continue
        
        # Clean up whitespace
        comp, defect = comp.strip(), defect.strip()
        count = 0
        
        # Load all image files from the folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({"path": os.path.join(folder_path, img_file), "component": comp, "defect": defect})
                count += 1
        
        # Track sample count per category
        if count > 0:
            summary[f"{comp}_{defect}"] = count

    # Print statistics
    total_images = sum(summary.values())
    print("\nImage Summary by Component-Defect Pair:")
    for k, v in sorted(summary.items()):
        percent = (v / total_images) * 100 if total_images > 0 else 0
        print(f"{k}: {v} images ({percent:.2f}%)")
    print(f"\nTotal Images Loaded: {total_images}")

    # Visualize dataset distribution
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
# Initialize classifier (model loaded lazily on first prediction)
clf = TwoDimImageClassifier()

@app.get("/")
def root():
    """Health check and API info endpoint."""
    return {"message": "Component-Defect Classifier API is running. Use /predict or /health."}

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {"status": "UP", "model_version": "component-defect-v1"}


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Predict component and defect for uploaded image.
    
    Returns: JSON with format "component_defect_confidence"
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Get prediction
        result = clf.predict(temp_path)
        # Clean up temporary file
        os.remove(temp_path)
        
        # Format response
        formatted = f"{result['component']}_{result['defect']}_{result['confidence']['average_confidence']:.2f}"
        return {"prediction": formatted}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

import socket

def find_available_port(preferred_ports=[8080, 8000, 8500]):
    """
    Find an available port from the preferred list.
    
    Args:
        preferred_ports: List of ports to try in order
        
    Returns:
        Available port number or 8500 (fallback)
    """
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try to bind to the port
                s.bind(("0.0.0.0", port))
                s.close()
                return port
            except OSError:
                # Port in use, try next
                continue
    return 8500  # Default fallback

if __name__ == "__main__":
    image_root = r"C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\images"
    print("Loading labeled samples from:", image_root)

    # Quick Mode: Skip training if model checkpoint exists
    if os.path.exists(clf.ckpt_path):
        print("Quick Mode: Model checkpoint found. Skipping training...")
        clf._load_checkpoint()
    else:
        # Load training data
        samples = load_labeled_samples(image_root)
        print(f"Found {len(samples)} labeled images for training.")
        # Train model if samples available
        if len(samples) > 0:
            clf.train(samples, epochs=10, batch_size=8, freeze_backbone=True)
            print("Training complete.")
        else:
            print("No labeled images found for training.")

    # Start API server on available port
    port = find_available_port()
    print(f"Starting prediction server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
