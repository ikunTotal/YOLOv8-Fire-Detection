import os, shutil, random
random.seed(0)
import numpy as np
from sklearn.model_selection import train_test_split

val_size = 0.1
test_size = 0.1
postfix = 'jpg'
imgpath = '/root/lanyun-tmp/firedetn/images'
txtpath = '/root/lanyun-tmp/firedetn/labels'

os.makedirs('/root/lanyun-tmp/firedetn/train/images', exist_ok=True)
os.makedirs('/root/lanyun-tmp/firedetn/val/images', exist_ok=True)
os.makedirs('/root/lanyun-tmp/firedetn/test/images', exist_ok=True)
os.makedirs('/root/lanyun-tmp/firedetn/train/labels', exist_ok=True)
os.makedirs('/root/lanyun-tmp/firedetn/val/labels', exist_ok=True)
os.makedirs('/root/lanyun-tmp/firedetn/test/labels', exist_ok=True)

listdir = np.array([i for i in os.listdir(txtpath) if 'txt' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/root/lanyun-tmp/firedetn/train/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/root/lanyun-tmp/firedetn/train/labels/{}'.format(i))

for i in val:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/root/lanyun-tmp/firedetn/val/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/root/lanyun-tmp/firedetn/val/labels/{}'.format(i))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/root/lanyun-tmp/firedetn/test/images/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/root/lanyun-tmp/firedetn/test/labels/{}'.format(i))