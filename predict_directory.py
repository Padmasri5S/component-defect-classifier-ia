import os
from two_dim_classifier import TwoDimImageClassifier  # Import your previous class

# ----------------------------
# Configuration
# ----------------------------
image_dir = r"C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\images"
components = [
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support"
]
defects = [
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose",
    "Missing", "Drummy", "Leaks"
]

# Initialize classifier
clf = TwoDimImageClassifier(components, defects)

# ----------------------------
# Predict for all images
# ----------------------------
print(f"Scanning directory: {image_dir}")
for img_name in os.listdir(image_dir):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(image_dir, img_name)
        result = clf.predict(img_path)

        # Extract confidence details
        conf_struct = result["confidence"]["head1"]
        conf_defect = result["confidence"]["head2"]
        avg_conf = (max(conf_struct.values()) + max(conf_defect.values())) / 2.0

        print("\n----------------------------------------")
        print(f"Image: {img_name}")
        print(f"Predicted Component: {result['label1']} (Confidence: {max(conf_struct.values()):.4f})")
        print(f"Predicted Defect:    {result['label2']} (Confidence: {max(conf_defect.values()):.4f})")
        print(f"Average Confidence:  {avg_conf:.4f}")
        print("Confidence breakdown:")
        print(f"  Components: {conf_struct}")
        print(f"  Defects:    {conf_defect}")
        print("Note: Confidence scores are probabilities (0.0–1.0). Average confidence = mean of top predictions for both heads.")