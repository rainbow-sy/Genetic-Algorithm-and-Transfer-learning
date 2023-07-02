import numpy as np
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
#from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras import Model
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
#from keras.models import Model
#from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
from DataMgr import load_cifar10, write_performance
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
#数据集
#root = 'D:/datasets/cifar10'
root = r'E:\遗传算法_迁移学习 -VGG\数据集\cifar10'
# root = '/home/u800199/workdir/datasets/cifar10'
X_train, y_train, X_test, y_test = load_cifar10(root)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# X, Y = load_data(root)
# X_train, X_test = X[0:80], X[80:100]
# y_train, y_test = Y[0:80], Y[80:100]
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#创建模型
base_model = VGG16(weights='imagenet', include_top=False)  # 不包含最顶层
# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 添加一个全连接层
x = Dense(1024, activation='relu')(x)
# 添加一个分类器，假设我们有200个类
predictions = Dense(10, activation='softmax')(x)
# 构建我们需要训练的完整模型
model_type = Model(inputs=base_model.input, outputs=predictions)


#固定训练层数
model=model_type
for layer in model.layers[:21]:
    layer.trainable = False
for layer in model.layers[21:]:
    layer.trainable = True
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# train_loss, train_acc = model.evaluate(_X, _y, verbose=0)
# model.fit(X_train, y_train,
#                       epochs=25,
#                       batch_size=32)
# test_loss, test_acc= model.evaluate(X_test, y_test, batch_size=32)

history = model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test), validation_freq=1,
                    )
print('训练集准确率：',history.history['accuracy'][-1])
print('验证集准确率：',history.history['val_accuracy'][-1])
###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

