#%%
import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from datetime import datetime

current_time = datetime.now()

#%%
PED_MODEL_PATH = './runs/train/exp72/weights/best.pt'
CROSS_MODEL_PATH = './runs/train/exp70/weights/best.pt'

TEST_VIDEO_PATH = '../test-video/'
TEST_VIDEO_SAVE_PATH = TEST_VIDEO_PATH + 'output/'
CURRENT_TIME = f'{current_time.year}-{current_time.month}-{current_time.day} {current_time.hour}:{current_time.minute}:{current_time.second}'

TEST_VIDEO = 'hospital2.mp4'
SAVE_VIDEO = CURRENT_TIME + '.mp4'

#%%
img_size = 640
conf_thres = 0.1
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False

#%%
print('[Car, pedestrain model info]')
ped_device = torch.device('cpu')
print(ped_device)
ped_ckpt = torch.load(PED_MODEL_PATH, map_location = ped_device)
ped_model = ped_ckpt['ema' if ped_ckpt.get('cma') else 'model'].float().fuse().eval()
ped_class_names = ['보행자', '차량']
ped_stride = int(ped_model.stride.max())
ped_colors = ((50, 50, 50), (255, 0, 0))
print('\n')

#%%
print('[Cross, traffic light model info]')
cross_device = torch.device('cpu')
print(cross_device)
cross_ckpt = torch.load(CROSS_MODEL_PATH, map_location = cross_device)
cross_model = cross_ckpt['ema' if cross_ckpt.get('cma') else 'model'].float().fuse().eval()
cross_class_names = ['횡단보도', '빨간불', '초록불']
cross_stride = int(cross_model.stride.max())
cross_colors = ((255, 0, 255), (0, 0, 255), (0, 255, 0))
print('\n')

#%%
cap = cv2.VideoCapture(TEST_VIDEO_PATH + TEST_VIDEO)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(TEST_VIDEO_SAVE_PATH + SAVE_VIDEO, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# %%
# input image processing
def img_process(img, stride, device):
  img_input = letterbox(img, img_size, stride = stride)[0]
  img_input = img_input.transpose((2, 0, 1))[::-1]
  img_input = np.ascontiguousarray(img_input)
  img_input = torch.from_numpy(img_input).to(device)
  img_input = img_input.float()
  img_input /= 255.
  img_input = img_input.unsqueeze(0)

  return img_input

#%%
# predict classes
def pred_classes(pred, class_names:list, ignore_class_names:list, annotator, colors)->dict:
  assert class_names == ped_class_names or class_names == cross_class_names, 'given class names are not allowed'

  preds = {class_name:[] for class_name in class_names if class_name not in ignore_class_names}

  for p in pred:
    class_name = class_names[int(p[5])]
    # x1, y1, x2, y2
    position = p[:4]

    if class_name not in ignore_class_names:
      preds[class_name].append(position)
      if annotator is not None:
        annotator.box_label(position, '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

  return preds

#%%
def detect(img, stride, device, model, class_names, ignore_class_names, colors, annotator=None):
    global cx1, cy1, cx2, cy2

    img_input = img_process(img, stride, device)

    pred = model(img_input, augment = False, visualize = False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det)[0]
    pred = pred.cpu().numpy()

    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()
    preds = pred_classes(pred, class_names, ignore_class_names, annotator, colors)

    return preds

#%%
print('[Check the presence of traffic light]')
isTrafficLight = False

while True:
  print('Is there traffic light? [y/n]')
  ans = input()
  if ans == 'y' or ans == 'Y':
    isTrafficLight = True
    break
  elif ans == 'n' or ans == 'N':
    isTrafficLight = False
    break
  else:
    print("Please enter a valid key")
print('\n')

#%%
print('[Check the pedestrain safety range]')
_, img = cap.read()

cnt = 0
isClick = False
isFinish = False

safe_x1, safe_y1 = 0, 0
safe_x2, safe_y2 = 0, 0

def click_event(event, x, y, flags, param):
    global cnt, isClick, isFinish
    global safe_x1, safe_y1, safe_x2, safe_y2
    if isFinish:
        return

    if isClick is False:
        if cnt == 1:
            print('Click the right down position')
        elif cnt == 2:
            print('Click the upper position')
        elif cnt == 3:
            print('Finish... Please press the \'0\'')
            isFinish = True
            return
        isClick = True
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        if cnt == 0:
            safe_x1 = x
        elif cnt == 1:
            safe_x2 = x
        elif cnt == 2:
            safe_y1, safe_y2 = y, y
        cnt += 1
        isClick = False

print('Click the left down position')

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('safe_x1, safe_y1, safe_x2, safe_y2 = ', safe_x1, safe_y1, safe_x2, safe_y2)
print('\n')

#%%
print('[Run the model]')
# 횡단보도 찾고 고정
ret, img = cap.read()
peds_dict = detect(img, cross_stride, cross_device, cross_model, cross_class_names, ['차량'],cross_colors)
cx1, cy1, cx2, cy2 = peds_dict['횡단보도'][0]

#%%
cur_frame = 1
pbar = tqdm(total=frames)

green = cv2.imread('./green.png')
yellow = cv2.imread('./yellow.png')
red = cv2.imread('./red.png')

rows, cols, _ = green.shape
H, W, _ = img.shape
start_row = int(H / 45)
start_col = int(W / 2) - int(cols / 2)

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  pbar.update(cur_frame)

  annotator = Annotator(img.copy(), line_width = 3, example = '한글', font = 'data/malgun.ttf')

  peds_dict = detect(img, ped_stride, ped_device, ped_model, ped_class_names, ['차량'], ped_colors, annotator)

  ignore_list = ['횡단보도'] if isTrafficLight is True else ['횡단보도', '초록불', '빨간불']
  light_dict = detect(img, cross_stride, cross_device, cross_model, cross_class_names, ignore_list, cross_colors, annotator)
  annotator.box_label([cx1, cy1, cx2, cy2], '횡단보도', color=(255, 0, 255))

  img = annotator.result()
  result_img = img.copy()
  cv2.rectangle(result_img, (int(safe_x1), int(safe_y1)), (int(safe_x2), int(cy1)), (255, 255, 255), 3)

  peds = peds_dict['보행자']

  cross_light_color = ('초록불' if len(light_dict['초록불']) else '빨간불') if isTrafficLight is True else None

  # safety 체크 알고리즘
  in_safety, in_cross = False, False

  if len(peds):
    for ped in peds:
      px1, py1, px2, py2 = ped

      _in_safety = int(safe_y1) <= int(py2) <= int(cy1) and int(safe_x1) <= int(px1) and int(px2) <= int(safe_x2)
      _in_cross = int(cy1) <= int(py2) <= int(cx2) and int(cx1) <= int(px1) and int(px2) <= int(cx2)
      in_safety, in_cross = in_safety or _in_safety, in_cross or _in_cross

    # red : stop!; yellow : stop and go; green : drive slowly
    if in_cross:
      result_img[start_row:start_row+rows, start_col:start_col+cols] = red
    elif in_safety:
      if cross_light_color == None:
        result_img[start_row:start_row+rows, start_col:start_col+cols] = red
      elif cross_light_color == '초록불':
        result_img[start_row:start_row+rows, start_col:start_col+cols] = red
      else:
        result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
    else:   
      if cross_light_color == None:
        result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
      elif cross_light_color == '초록불':
        result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
      else:
        result_img[start_row:start_row+rows, start_col:start_col+cols] = green
  else:
    # no ped
    result_img[start_row:start_row+rows, start_col:start_col+cols] = green

  out.write(result_img)
  if cv2.waitKey(1) == ord('q'):
    break

#%%
cap.release()
out.release()

# %%
