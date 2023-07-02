from keras.layers import Dropout, Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet import ResNet50
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from Dataloader import load_data

#数据集
X_train, y_train, X_test, y_test = load_data()
y_train, y_test = to_categorical(y_train,4), to_categorical(y_test,4)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

net = ResNet50(
    weights='imagenet', # Load weights pre-trained on ImageNet.
     include_top=False # Do not include the ImageNet classifier at the top.
     )
model = net.output
model = GlobalAveragePooling2D()(model)
model = Dropout(0.4)(model)
model = Dense(4, activation="softmax")(model)
model = Model(inputs= net.input, outputs= model)

#固定训练层数
# model=model_type
for layer in model.layers[:280]:
    layer.trainable = False
for layer in model.layers[280:]:
    layer.trainable = True

#compile our model.
adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=25, validation_data=(X_test, y_test), validation_freq=1,
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










