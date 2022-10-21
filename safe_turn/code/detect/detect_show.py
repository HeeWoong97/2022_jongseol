#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '"/content/drive/MyDrive/Colab Notebooks/yolov5"')


# In[3]:


import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from tqdm import tqdm


# In[24]:


# PED_MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/yolov5/runs/train/exp18/weights/best.pt'
# CROSS_MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/crosswalk-traffic-light-detection-yolov5-master/runs/train/exp4/weights/best.pt'
MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/yolov5/runs/train/exp68/weights/best.pt'
TEST_VIDEO_PATH = '/content/drive/MyDrive/Colab Notebooks/test-video/'
TEST_VIDEO_SAVE_PATH = TEST_VIDEO_PATH + 'output/'

img_size = 640
conf_thres = 0.1
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False


# In[ ]:


# ped_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(ped_device)
# ped_ckpt = torch.load(PED_MODEL_PATH, map_location = ped_device)
# ped_model = ped_ckpt['ema' if ped_ckpt.get('cma') else 'model'].float().fuse().eval()
# ped_class_names = ['보행자', '차량']
# ped_stride = int(ped_model.stride.max())
# ped_colors = ((0, 0, 255), (0, 255, 0))


# In[ ]:


# cross_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(cross_device)
# cross_ckpt = torch.load(CROSS_MODEL_PATH, map_location = cross_device)
# cross_model = cross_ckpt['ema' if cross_ckpt.get('cma') else 'model'].float().fuse().eval()
# cross_class_names = ['횡단보도', '빨간불', '초록불']
# cross_stride = int(cross_model.stride.max())
# cross_colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0))


# In[25]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
ckpt = torch.load(MODEL_PATH, map_location = device)
model = ckpt['ema' if ckpt.get('cma') else 'model'].float().fuse().eval()
class_names = ['보행자', '차량', '횡단보도', '빨간불', '초록불']
stride = int(model.stride.max())
colors = ((50, 50, 50), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0))


# In[26]:


cap = cv2.VideoCapture(TEST_VIDEO_PATH + 'ewha.mp4')
# cap = cv2.VideoCapture(TEST_VIDEO_SAVE_PATH + 'output10.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(TEST_VIDEO_SAVE_PATH + 'output16.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# In[27]:


cur_frame = 1
pbar = tqdm(total=frames)

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  pbar.update(cur_frame)

  H, W, _ = img.shape

  img_input = letterbox(img, img_size, stride = stride)[0]
  img_input = img_input.transpose((2, 0, 1))[::-1]
  img_input = np.ascontiguousarray(img_input)
  img_input = torch.from_numpy(img_input).to(device)
  img_input = img_input.float()
  img_input /= 255.
  img_input = img_input.unsqueeze(0)

  pred = model(img_input, augment = False, visualize = False)[0]

  pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det)[0]

  pred = pred.cpu().numpy()

  pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()

  annotator = Annotator(img.copy(), line_width = 3, example = str(class_names), font = 'data/malgun.ttf')

  for p in pred:
    class_name = class_names[int(p[5])]
    x1, y1, x2, y2 = p[:4]

    annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

  result_img = annotator.result()

  out.write(result_img)
  if cv2.waitKey(1) == ord('q'):
    break


# In[28]:


cap.release()
out.release()


# In[ ]:




