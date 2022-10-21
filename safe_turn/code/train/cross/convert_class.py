#!/usr/bin/env python
# coding: utf-8

# In[41]:


from google.colab import drive
drive.mount('/content/drive')


# In[42]:


import os
from glob import glob
from tqdm import tqdm


# In[43]:


DATASET1_PATH = '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset1'
DATASET2_PATH = '/content/drive/MyDrive/Colab Notebooks/data/cross/dataset2'


# ### convert dataset1

# #### convert train

# In[44]:


TRAIN_LABEL_ORIGIN_PATH = os.path.join(DATASET1_PATH, 'train', 'labels_1')
TRAIN_LABEL_SAVE_PATH = os.path.join(DATASET1_PATH, 'train', 'labels')

txt_list = glob(os.path.join(TRAIN_LABEL_ORIGIN_PATH, '*.txt'))
print(len(txt_list))

# for txt_path in tqdm(txt_list):
#     edited_lines = []
#     file_name = txt_path.split('/')[-1]
#     txt_save_path = os.path.join(TRAIN_LABEL_SAVE_PATH, file_name)

#     with open(txt_path, 'r') as fr:
#         lines = fr.readlines()
        
#         for line in lines:
#             tokens = line.split(' ')
#             past_label = int(tokens[0])
#             new_label = past_label + 2
#             tokens[0] = str(new_label)
            
#             _edited_line = ' '.join(token for token in tokens)
#             edited_lines.append(_edited_line)
    
#     with open(txt_save_path, 'w') as fw:
#         fw.writelines(edited_lines)

txt_list = glob(os.path.join(TRAIN_LABEL_SAVE_PATH, '*.txt'))
print(len(txt_list))


# #### convert valid

# In[38]:


VALID_LABEL_ORIGIN_PATH = os.path.join(DATASET1_PATH, 'valid', 'labels_1')
VALID_LABEL_SAVE_PATH = os.path.join(DATASET1_PATH, 'valid', 'labels')

txt_list = glob(os.path.join(VALID_LABEL_ORIGIN_PATH, '*.txt'))
print(len(txt_list))

# for txt_path in tqdm(txt_list):
#     edited_lines = []
#     file_name = txt_path.split('/')[-1]
#     txt_save_path = os.path.join(VALID_LABEL_SAVE_PATH, file_name)
    
#     with open(txt_path, 'r') as fr:
#         lines = fr.readlines()
        
#         for line in lines:
#             tokens = line.split(' ')
#             past_label = int(tokens[0])
#             new_label = past_label + 2
#             tokens[0] = str(new_label)
            
#             _edited_line = ' '.join(token for token in tokens)
#             edited_lines.append(_edited_line)
    
#     with open(txt_save_path, 'w') as fw:
#         fw.writelines(edited_lines)

txt_list = glob(os.path.join(VALID_LABEL_SAVE_PATH, '*.txt'))
print(len(txt_list))


# ### convert dataset2

# #### convert train

# In[39]:


TRAIN_LABEL_ORIGIN_PATH = os.path.join(DATASET2_PATH, 'train', 'labels_1')
TRAIN_LABEL_SAVE_PATH = os.path.join(DATASET2_PATH, 'train', 'labels')

txt_list = glob(os.path.join(TRAIN_LABEL_ORIGIN_PATH, '*.txt'))
print(len(txt_list))

# for txt_path in tqdm(txt_list):
#     edited_lines = []
#     file_name = txt_path.split('/')[-1]
#     txt_save_path = os.path.join(TRAIN_LABEL_SAVE_PATH, file_name)
    
#     with open(txt_path, 'r') as fr:
#         lines = fr.readlines()
        
#         for line in lines:
#             tokens = line.split(' ')
#             past_label = int(tokens[0])
#             new_label = past_label + 2
#             tokens[0] = str(new_label)
            
#             _edited_line = ' '.join(token for token in tokens)
#             edited_lines.append(_edited_line)
    
#     with open(txt_save_path, 'w') as fw:
#         fw.writelines(edited_lines)

txt_list = glob(os.path.join(TRAIN_LABEL_SAVE_PATH, '*.txt'))
print(len(txt_list))


# #### convert valid

# In[40]:


VALID_LABEL_ORIGIN_PATH = os.path.join(DATASET2_PATH, 'valid', 'labels_1')
VALID_LABEL_SAVE_PATH = os.path.join(DATASET2_PATH, 'valid', 'labels')

txt_list = glob(os.path.join(VALID_LABEL_ORIGIN_PATH, '*.txt'))
print(len(txt_list))

# for txt_path in tqdm(txt_list):
#     edited_lines = []
#     file_name = txt_path.split('/')[-1]
#     txt_save_path = os.path.join(VALID_LABEL_SAVE_PATH, file_name)
    
#     with open(txt_path, 'r') as fr:
#         lines = fr.readlines()
        
#         for line in lines:
#             tokens = line.split(' ')
#             past_label = int(tokens[0])
#             new_label = past_label + 2
#             tokens[0] = str(new_label)
            
#             _edited_line = ' '.join(token for token in tokens)
#             edited_lines.append(_edited_line)
    
#     with open(txt_save_path, 'w') as fw:
#         fw.writelines(edited_lines)

txt_list = glob(os.path.join(VALID_LABEL_SAVE_PATH, '*.txt'))
print(len(txt_list))


# In[ ]:




