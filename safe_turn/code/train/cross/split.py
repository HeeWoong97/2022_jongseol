#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
from glob import glob
import json
from shutil import copy2, move
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[ ]:


TRAIN_PATH_1 = "/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1/train/"
TEST_PATH_1 = "/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1/valid/"

TRAIN_IMG_1 = TRAIN_PATH_1 + "images/"
TEST_IMG_1 = TEST_PATH_1 + "images/"
TRAIN_LABEL_1 = TRAIN_PATH_1 + "labels/"
TEST_LABEL_1 = TEST_PATH_1 + "labels/"

##

TRAIN_PATH_2 = "/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2/train/"
TEST_PATH_2 = "/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2/valid/"

TRAIN_IMG_2 = TRAIN_PATH_2 + "images/"
TEST_IMG_2 = TEST_PATH_2 + "images/"
TRAIN_LABEL_2 = TRAIN_PATH_2 + "labels/"
TEST_LABEL_2 = TEST_PATH_2 + "labels/"


# ### train split

# In[ ]:


jpg_list = glob(os.path.join(TRAIN_IMG_1, '*.jpg'))
png_list = glob(os.path.join(TRAIN_IMG_1, '*.png'))
train_imgs = jpg_list + png_list

print(len(jpg_list))
print(len(png_list))
print(len(train_imgs))


# In[ ]:


train_labels = glob(os.path.join(TRAIN_LABEL_1, '*.txt'))

print(len(train_labels))


# In[ ]:


split_size = len(train_imgs) // 2 if len(train_imgs) < len(train_labels) else len(train_labels) // 2

split_imgs = train_imgs[:split_size]

for file_path in tqdm(split_imgs):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(TRAIN_LABEL_1 + label_name):
    move(TRAIN_IMG_1+img_name, TRAIN_IMG_2+img_name)
    move(TRAIN_LABEL_1+label_name, TRAIN_LABEL_2+label_name)
  else:
    print(img_name)


# In[ ]:


print(split_size, len(train_labels), len(train_imgs))


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1')


# In[ ]:


train_jpg_imgs = glob(os.path.join("train", "images", "*.jpg"))
train_png_imgs = glob(os.path.join("train", "images", "*.png"))
train_imgs = train_jpg_imgs + train_png_imgs

print(len(train_jpg_imgs))
print(len(train_png_imgs))
print(len(train_imgs))


# In[ ]:


with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_imgs) + '\n')


# #### save train.txt

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2')


# In[ ]:


train_jpg_imgs = glob(os.path.join("train", "images", "*.jpg"))
train_png_imgs = glob(os.path.join("train", "images", "*.png"))
train_imgs = train_jpg_imgs + train_png_imgs

print(len(train_jpg_imgs))
print(len(train_png_imgs))
print(len(train_imgs))


# In[ ]:


with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_imgs) + '\n')


# ### test split

# In[ ]:


jpg_list = glob(os.path.join(TEST_IMG_1, '*.jpg'))
png_list = glob(os.path.join(TEST_IMG_1, '*.png'))
test_imgs = jpg_list + png_list

print(len(jpg_list))
print(len(png_list))
print(len(test_imgs))


# In[ ]:


test_labels = glob(os.path.join(TEST_LABEL_1, '*.txt'))

print(len(test_labels))

split_size = 3515

print(split_size)

split_imgs = test_imgs[:split_size]
print(len(split_imgs))


# In[ ]:


# split_size = len(test_imgs) // 2 if len(test_imgs) < len(test_labels) else len(test_labels) // 2

# split_imgs = test_imgs[:split_size]

for file_path in tqdm(split_imgs):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(TEST_LABEL_1 + label_name):
    move(TEST_IMG_1+img_name, TEST_IMG_2+img_name)
    move(TEST_LABEL_1+label_name, TEST_LABEL_2+label_name)
  else:
    print(img_name)


# In[ ]:


print(split_size, len(test_labels), len(test_imgs))


# #### save valid.txt

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1')


# In[ ]:


test_jpg_imgs = glob(os.path.join("valid", "images", "*.jpg"))
test_png_imgs = glob(os.path.join("valid", "images", "*.png"))
test_imgs = test_jpg_imgs + test_png_imgs

print(len(test_jpg_imgs))
print(len(test_png_imgs))
print(len(test_imgs))


# In[ ]:


with open('valid.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_imgs) + '\n')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2')


# In[ ]:


test_jpg_imgs = glob(os.path.join("valid", "images", "*.jpg"))
test_png_imgs = glob(os.path.join("valid", "images", "*.png"))
test_imgs = test_jpg_imgs + test_png_imgs

print(len(test_jpg_imgs))
print(len(test_png_imgs))
print(len(test_imgs))


# In[ ]:


with open('valid.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_imgs) + '\n')


# ### restore files

# In[ ]:


# img_list = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2/valid/images", "*.jpg"))
# print(len(img_list))
# print(img_list[0])
# for img in tqdm(img_list):
#   move(img, os.path.join("/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1/valid/images", img.split('/')[-1]))


# In[ ]:


# txt_list = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2/valid/labels", "*.txt"))
# print(len(txt_list))
# print(txt_list[0])
# for txt in tqdm(txt_list):
#   move(txt, os.path.join("/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1/valid/labels", txt.split('/')[-1]))


# In[ ]:




