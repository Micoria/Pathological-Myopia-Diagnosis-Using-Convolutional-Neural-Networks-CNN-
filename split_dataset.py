import os
import pandas as pd
# classified labels for training images
train_dataset = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Training400/PALM-Training400"
train_list = []
label_list = []


train_filenames = os.listdir(train_dataset)

for name in train_filenames:
    filepath = os.path.join(train_dataset, name)
    train_list.append(filepath)
    if name[0] == 'N' or name[0] == 'H':
        label = 0
        label_list.append(label)
    elif name[0] == 'P':
        label = 1
        label_list.append(label)
    else:
        raise('Error dataset!')


with open('/Users/micoria/Desktop/SqueezeNet/train.txt', 'w', encoding='UTF-8') as f:
    i = 0
    for train_img in train_list:
        f.write(str(train_img) + ' ' +str(label_list[i]))
        i += 1
        f.write('\n')
# # classified labels for validating images
valid_dataset = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Validation400"
valid_filenames = os.listdir(valid_dataset)
valid_label = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx"
data = pd.read_excel(valid_label)
valid_data = data[['imgName', 'Label']].values.tolist()

with open('/Users/micoria/Desktop/SqueezeNet/valid.txt', 'w', encoding='UTF-8') as f:
    for valid_img in valid_data:
        f.write(str(valid_dataset) + '/' + valid_img[0] + ' ' + str(valid_img[1]))
        f.write('\n')