from PIL import Image
import numpy as np
from six.moves import cPickle as pickle

def load_data(root):
    file = root + '/test_batch'
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        X = d['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')#返回形状10000，32，32，3
        y = np.array(d['labels'])
    X=X[0:100]
    y=y[0:100]
    X_resized = np.zeros((100,229,229,3))# 创建一个存储修改过图片分辨率的矩阵
    for i in range(0,100):
        img = X[i]
        img = Image.fromarray(np.uint8(img))
        img = np.array(img.resize((229,229),Image.BICUBIC))# 修改分辨率，再转为array类
        X_resized[i,:,:,:] = img
    X_resized /= 255
    return X_resized,y