from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
# 生成虚拟数据
import numpy as np
x_train = np.random.random((1000, 20))
y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          epochs=3,
          batch_size=128)
#返回一个字典，包含所有epochs的损失和准确率
#{'loss': [2.3, 2.3, 2.3], 'accuracy': [0.08, 0.09, 0.09]}
score = model.evaluate(x_test, y_test, batch_size=128)  #2.3039, 0.1099,损失和准确率
print(history.history)
print(score)
print(history.history['accuracy'][-1])
# history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=3)
# #{'loss': [2.3, 2.3, 2.3, 2.3, 2.2], 'accuracy': [0.1, 0.1, 0.1, 0.1, 0.1], 'val_loss': [2.3, 2.2, 2.2, 2.2, 2.2], 'val_accuracy': [0.1, 0.1, 0.1, 0.1, 0.1]}
# print(history.history)
# print(history.history['val_loss'])
# import keras
# adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# lista=[1,2,2,5]
# print(lista[-1])