#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
from glob import glob
import json
from shutil import copy2, move
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[3]:


class_names = ['Zebra_Cross', 'R_Signal', 'G_Signal']


# In[4]:


ORIGIN_PATH = "/content/drive/MyDrive/Colab Notebooks/data/cross/origin/"
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/data/cross/dataset/"


# In[13]:


jpg_list = glob(os.path.join(ORIGIN_PATH, '*', '*', '*.jpg'))

len(jpg_list)


# In[14]:


png_list = glob(os.path.join(ORIGIN_PATH, '*', '*', '*.png'))

len(png_list)


# In[19]:


img_list = jpg_list + png_list

len(img_list)
img_list


# In[14]:


os.makedirs(os.path.join(DATA_PATH, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, 'valid', 'images'), exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, 'valid', 'labels'), exist_ok=True)


# In[ ]:


train_img_list, val_img_list = train_test_split(img_list, test_size=0.4, random_state=2022)

len(train_img_list), len(val_img_list)


# In[21]:


file_list = []

for img_path in tqdm(img_list):
    json_path = img_path.replace('.jpg', '.json') if img_path.find('.jpg') is not -1 else img_path.replace('.png', '.json')
    # json_path = img_path.replace('.jpg', '.json')
    print(json_path)

    try:
      with open(json_path, 'r') as f:
          data = json.load(f)
    except Exception as e:
      continue

    w = data['imageWidth']
    h = data['imageHeight']
    
    txt = ''
    
    try:
        for shape in data['shapes']:
            label = shape['label']

            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            cx = (x1 + x2) / 2. / w
            cy = (y1 + y2) / 2. / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label = class_names.index(shape['label'])

            txt += '%d %f %f %f %f\n' % (label, cx, cy, bw, bh)

        move(img_path, os.path.join(DATA_PATH, 'train', 'images', os.path.basename(img_path)))

        with open(os.path.join(DATA_PATH, 'train', 'labels', os.path.basename(json_path).replace('.json', '.txt')), 'w') as f:
            f.write(txt)
        
        file_list.append(os.path.join('train', 'images', os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)
    
with open(os.path.join(DATA_PATH, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
        
print(len(file_list))


# In[ ]:


file_list = []

for img_path in tqdm(val_img_list):
    json_path = img_path.replace('.jpg', '.json') if img_path.find('.jpg') is not -1 else img_path.replace('.png', '.json')
    # json_path = img_path.replace('.jpg', '.json')

    try:
      with open(json_path, 'r') as f:
          data = json.load(f)
    except Exception as e:
      continue

    w = data['imageWidth']
    h = data['imageHeight']
    
    txt = ''
    
    try:
        for shape in data['shapes']:
            label = shape['label']

            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            cx = (x1 + x2) / 2. / w
            cy = (y1 + y2) / 2. / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label = class_names.index(shape['label'])

            txt += '%d %f %f %f %f\n' % (label, cx, cy, bw, bh)

        move(img_path, os.path.join(DATA_PATH, 'valid', 'images', os.path.basename(img_path)))

        with open(os.path.join(DATA_PATH, 'valid', 'labels', os.path.basename(json_path).replace('.json', '.txt')), 'w') as f:
            f.write(txt)
        
        file_list.append(os.path.join('valid', 'images', os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)
    
with open(os.path.join(DATA_PATH, 'valid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
        
print(len(file_list))


# In[ ]:




