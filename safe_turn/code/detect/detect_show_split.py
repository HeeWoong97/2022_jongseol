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


# In[4]:


PED_MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/yolov5/runs/train/exp72/weights/best.pt'
CROSS_MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/yolov5/runs/train/exp70/weights/best.pt'

TEST_VIDEO_PATH = '/content/drive/MyDrive/Colab Notebooks/test-video/'
TEST_VIDEO_SAVE_PATH = TEST_VIDEO_PATH + 'output/'

img_size = 640
conf_thres = 0.5
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False


# In[5]:


ped_device = torch.device('cpu')
print(ped_device)
ped_ckpt = torch.load(PED_MODEL_PATH, map_location = ped_device)
ped_model = ped_ckpt['ema' if ped_ckpt.get('cma') else 'model'].float().fuse().eval()
ped_class_names = ['보행자', '차량']
ped_stride = int(ped_model.stride.max())
ped_colors = ((50, 50, 50), (255, 0, 0))


# In[6]:


cross_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(cross_device)
cross_ckpt = torch.load(CROSS_MODEL_PATH, map_location = cross_device)
cross_model = cross_ckpt['ema' if cross_ckpt.get('cma') else 'model'].float().fuse().eval()
cross_class_names = ['횡단보도', '빨간불', '초록불']
cross_stride = int(cross_model.stride.max())
cross_colors = ((255, 0, 255), (0, 0, 255), (0, 255, 0))


# In[7]:


cap = cv2.VideoCapture(TEST_VIDEO_PATH + 'ewha.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(TEST_VIDEO_SAVE_PATH + 'output21.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# In[8]:


def detect(annotator, img, stride, device, model, class_names, colors):
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

    for p in pred:
        class_name = class_names[int(p[5])]
        x1, y1, x2, y2 = p[:4]

        annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])


# In[9]:


cur_frame = 1
pbar = tqdm(total=frames)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    pbar.update(cur_frame)

    annotator = Annotator(img.copy(), line_width = 3, example = '한글', font = 'data/malgun.ttf')

    detect(annotator, img, ped_stride, ped_device, ped_model, ped_class_names, ped_colors)
    detect(annotator, img, cross_stride, cross_device, cross_model, cross_class_names, cross_colors)

    result_img = annotator.result()

    out.write(result_img)
    if cv2.waitKey(1) == ord('q'):
        break


# In[10]:


cap.release()
out.release()


# In[ ]:




