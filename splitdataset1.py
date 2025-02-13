import os
import pandas as pd

# **training dataset: get labels from the name of images**
train_dataset = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Training400/PALM-Training400"
train_list = []
label_list = []

train_filenames = os.listdir(train_dataset)

for name in train_filenames:
    filepath = os.path.join(train_dataset, name)
    train_list.append(filepath)
    if name[0] == 'N' or name[0] == 'H':  # not Pathological myopia
        label = 0
        label_list.append(label)
    elif name[0] == 'P':  # Pathological myopia
        label = 1
        label_list.append(label)
    else:
        raise ValueError(f"Error dataset: {name}!")

# **write train.txt**
with open('/Users/micoria/Desktop/SqueezeNet/train.txt', 'w', encoding='UTF-8') as f:
    for i in range(len(train_list)):
        f.write(f"{train_list[i]} {label_list[i]}\n")
    f.flush()

# **validation dataset: get the labels from excel**
valid_dataset = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Validation400"
valid_label = "/Users/micoria/Desktop/SqueezeNet/data/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx"

data = pd.read_excel(valid_label)
valid_data = data[['imgName', 'Label']].values.tolist()

# **write valid.txt**
with open('/Users/micoria/Desktop/SqueezeNet/valid.txt', 'w', encoding='UTF-8') as f:
    for valid_img in valid_data:
        f.write(f"{valid_dataset}/{valid_img[0]} {valid_img[1]}\n")
    f.flush()

print("✅ `train.txt` 和 `valid.txt` 生成完毕！")
