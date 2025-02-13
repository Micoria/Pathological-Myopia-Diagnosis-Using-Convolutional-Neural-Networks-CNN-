from PIL import Image
import torch
from torchvision import transforms
from model import SqueezeNet

model = SqueezeNet(num_classes=2)
model.load_state_dict(torch.load("/Users/micoria/Desktop/SqueezeNet/best_model.pth", map_location=torch.device('cpu')))
model.eval()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = Image.open("/Users/micoria/Desktop/SqueezeNet/data/Test/PALM-Testing400-Images/T0008.jpg")
img = data_transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    predict = torch.softmax(output, dim=1)
    predict_class = torch.argmax(predict).item()
    confidence = predict[0][predict_class].item()

class_names = ['Non-pathological myopia', 'Pathological myopia']
print(f"✅ Prediction: {class_names[predict_class]}, Confidence: {confidence:.4f}")