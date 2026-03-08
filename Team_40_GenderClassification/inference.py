import torch
from torchvision import transforms, models
from PIL import Image
import os

class GenderModel:
    def __init__(self):
        # Re-build the MobileNetV3 Small structure
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 2)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(base_dir, 'model', 'model.pth')
        
        try:
            # Load weights to CPU only as per requirements
            self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_one(self, image_path):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, predicted_class = torch.max(probs, 0)
                
                # FINAL SUBMISSION MAPPING:
                # Based on requirements: 0 = Male, 1 = Female
                return predicted_class.item(), conf.item()
                    
        except Exception as e:
            return 0, 0.0

# Global instance for the judges to call
_model = GenderModel()

def predict(image_path):
    return _model.predict_one(image_path)