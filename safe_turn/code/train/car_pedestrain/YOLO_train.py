#!/usr/bin/env python
# coding: utf-8

# **Mount drive**

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import os
from glob import glob

train_jpg_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/train/images", "*.jpg"))
train_png_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/train/images", "*.png"))
train_img_list1 = train_jpg_list1 + train_png_list1
print("train img1: ", len(train_img_list1))
train_txt_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/train/labels", "*.txt"))
print("train txt1: ", len(train_txt_list1))
valid_jpg_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/valid/images", "*.jpg"))
valid_png_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/valid/images", "*.png"))
valid_img_list1 = valid_jpg_list1 + valid_png_list1
print("valid img1: ", len(valid_img_list1))
valid_txt_list1 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset1/valid/labels", "*.txt"))
print("valid txt1: ", len(valid_txt_list1))

print("\n")
train_jpg_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/train/images", "*.jpg"))
train_png_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/train/images", "*.png"))
train_img_list2 = train_jpg_list2 + train_png_list2
print("train img2: ", len(train_img_list2))
train_txt_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/train/labels", "*.txt"))
print("train txt2: ", len(train_txt_list2))
valid_jpg_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/valid/images", "*.jpg"))
valid_png_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/valid/images", "*.png"))
valid_img_list2 = valid_jpg_list2 + valid_png_list2
print("valid img2: ", len(valid_img_list2))
valid_txt_list2 = glob(os.path.join("/content/drive/MyDrive/Colab Notebooks/data/car_pedestrain/dataset2/valid/labels", "*.txt"))
print("valid txt2: ", len(valid_txt_list2))


# **Move directory**

# In[4]:


get_ipython().run_line_magic('cd', '"/content/drive/MyDrive/Colab Notebooks/yolov5"')


# **Install requirements**

# In[5]:


get_ipython().system('pip install -U -r requirements.txt')


# **Check torch version**

# In[6]:


import torch

print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# **Train dataset**

# In[7]:


get_ipython().system('python train.py --img 640 --batch 16 --epochs 30 --data car_pedestrain_data.yaml --weights yolov5s.pt --cache')


# In[ ]:




