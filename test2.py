import os
from PIL import Image
import torch
from torchvision import transforms
from model import SqueezeNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# **load model**
model = SqueezeNet(num_classes=2)
model.load_state_dict(torch.load("/Users/micoria/Desktop/SqueezeNet/best_model.pth", map_location=torch.device('cpu')))
model.eval()

# **define the proprocess of images**
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# **path of test dataset and test labels**
test_dir = "/Users/micoria/Desktop/SqueezeNet/data/Test/PALM-Testing400-Images"
label_file = "/Users/micoria/Desktop/SqueezeNet/data/Test/PM_Label_and_Fovea_Location.xlsx"

# **read labels**
df = pd.read_excel(label_file)
test_data = df[['imgName', 'Label']].values.tolist()

# **name of classes**
class_names = ['Non-pathological myopia', 'Pathological myopia']

# **save prediction and labels**
y_true = []
y_pred = []

# **loop**
for img_name, label in test_data:
    img_path = os.path.join(test_dir, img_name)

    # **check the existence of the iamges**
    if not os.path.isfile(img_path):
        print(f"⚠️ 文件不存在: {img_path}")
        continue

    # **save labels**
    y_true.append(label)

    # **prediction**
    img = Image.open(img_path)
    img = data_transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        predict = torch.softmax(output, dim=1)
        predict_class = torch.argmax(predict).item()
        confidence = predict[0][predict_class].item()

    y_pred.append(predict_class)
    print(f"🔍 {img_name} → Prediction: {class_names[predict_class]}, Confidence: {confidence:.4f}")

# **generate confusion matrix**
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Test Set")
plt.show()