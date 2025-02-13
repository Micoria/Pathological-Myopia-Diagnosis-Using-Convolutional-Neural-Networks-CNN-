import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from model import SqueezeNet
import torch


data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open("/Users/micoria/Desktop/SqueezeNet/data/Test/PALM-Testing400-Images/T0008.jpg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
name = ['Non-pathological myopia', 'Pathological myopia']
model_weight_path = "/Users/micoria/Desktop/SqueezeNet/best_model.pth"
model = SqueezeNet(num_classes=4)
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))

    predict = torch.softmax(output, dim=0)
    # Get the maximum likelihood index
    predict_cla = torch.argmax(predict).numpy()
    print('index is', predict_cla)
print('Prediction:{},Confidence: {}'.format(name[predict_cla], predict[predict_cla].item()))
plt.show()
