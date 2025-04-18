import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier.in_features, 1)
)


try:
    checkpoint = torch.load("model/best_model.pth", 
                          map_location=torch.device('cpu'),
                          weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

def predict_image(image, transform=None):
    """
    Make prediction on an image
    
    Args:
        image: PIL Image object
        transform: Optional transform to apply
    
    Returns:
        (prediction, probability)
    """
    if transform is None:
        transform = test_transform
    

    image_tensor = transform(image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    model.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).item()
        prediction = "Pneumonia" if probability > 0.5 else "Normal"
        
    return prediction, probability


image = Image.open(r"C:\Users\omarj\Desktop\X-ray\images\Pneumonia\person101_bacteria_483.jpeg").convert("RGB")
prediction, probability = predict_image(image)
print(f"Prediction: {prediction}, Probability: {probability:.4f}")