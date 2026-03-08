import torch
from torchvision import transforms, models
from PIL import Image
import os

class GenderModel:
    def __init__(self):
        print("🔄 Loading model...")

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features, 2
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(base_dir, "model", "model.pth")

        if not os.path.exists(weight_path):
            print("❌ model.pth not found at:", weight_path)
            exit()

        self.model.load_state_dict(
            torch.load(weight_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        print("✅ Model loaded successfully!")

    def predict_one(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, predicted_class = torch.max(probs, dim=1)

            # Flip labels (because training mapping was {'female': 0, 'male': 1})
            predicted_class = 1 - predicted_class

        return predicted_class.item(), conf.item()


_model = GenderModel()

def predict(image_path):
    return _model.predict_one(image_path)


if __name__ == "__main__":
    print("\n📷 Gender Classification Test")
    image_path = input("👉 Drag image here and press Enter:\n").strip().strip('"')

    if not os.path.exists(image_path):
        print("❌ File not found. Check path.")
    else:
        label, confidence = predict(image_path)
        gender = "Male" if label == 0 else "Female"

        print("\n✅ Prediction Result:")
        print("Gender:", gender)
        print("Confidence:", round(confidence * 100, 2), "%")

    input("\nPress Enter to exit...")