# --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------HANDWRITING RECOGNITION MODEL-----------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------
# LOADING AND VISUALISING DATASET
# from keras.datasets import mnist
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# for i in range(0, 6):
#     random_number = np.random.randint(0, len(x_train))
#     cv2.imshow(str(i), x_train[random_number])
#     cv2.waitKey()
# cv2.destroyAllWindows()
#
# plt.subplot(331)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.subplot(332)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.subplot(333)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.subplot(334)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.subplot(335)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.subplot(336)
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number], cmap=plt.get_cmap('gray'))
# plt.show()

# GETTING DATA IN  SHAPE TO TRAIN THE MODEL IN KERAS I.E. (NO OF SAMPLES, ROWS, COLS, DEPTH) FOR EG-> (6000, 28, 28, 1)
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
image_shape = (img_rows, img_cols, 1)
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')
x_train /= 255
x_test /= 255
print(x_test.shape)
print(x_train.shape)
print(x_test)

# HOT ONE ENCODING THE LABELS (OF OUR Y LABELS)
# (1,2,9,5,7)
# |--------------------------------------------|
# |\ /->|0   1   2   3   4   5   6   7   8   9 |
# |-----|--------------------------------------|
# |1    |0   1   0   0   0   0   0   0   0   0 |
# |2    |0   0   1   0   0   0   0   0   0   0 |
# |3    |0   0   0   0   0   0   0   0   0   1 |
# |4    |0   0   0   0   0   1   0   0   0   0 |
# |5    |0   0   0   0   0   0   0   1   0   0 |
# |.    |                                      |
# |.    |                                      |
# |.    |                                      |
# |--------------------------------------------|
from keras.utils import np_utils
y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)
# print(y_test)
num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]
print(str(num_pixels) + "<-pixels----classes->" + str(num_classes))

#BUILDING AND COMPILING THE MODEL

# import  keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras import backend as K
from  keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
print(model.summary())

# TRAINING THE MODEL

batch_size = 32
epochs = 1
history = model.fit(x_train,y_train,batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("test loss: " + str(score[0]))
print("test accuracy: " + str(score[1]))

# SAVING YOUR MODEL AT CUSTOM LOCATION
model.save('./HANDWRITINGRECOGNITION.h5') #<====LOCATION FOR FILE TO BE SAVED
print("MODEL SAVED")
# LOADING YOUR MODEL FROM CUSTOM LOCATION
# from keras.models import load_model
# model1 = load_model('./HANDWRITINGRECOGNITION.h5')


# PLOTING LOSS AND ACCURACY CHARTS

#<==========DOES NOT WORK=================================>
import matplotlib.pyplot as plt
# history_dict = history.history
# loss_val = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_val)+1)
# line1 = plt.plot(epochs, val_loss_values, label='Validation/Text Loss')
# line2 = plt.plot(epochs, loss_val, label='Training Loss')
# plt.setp(line1, linewidth=2.0, marker='.', markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker='v', markersize=10.0)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# acc_val = history_dict['acc']
# val_acc_values = history_dict['val_acc']
# epochs = range(1, len(acc_val)+1)
# line3 = plt.plot(epochs, val_acc_values, label='Validation/Text Accuracy')
# line4 = plt.plot(epochs, acc_val, label='Training Accuracy')
# plt.setp(line3, linewidth=2.0, marker='.', markersize=10.0)
# plt.setp(line4, linewidth=2.0, marker='v', markersize=10.0)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.legend()
# plt.show()
#<========================================================================>

# SAVING VISUAL REPRESENTATION OF OUT MODEL ( THIS TOOOOOO DOOES NOOOT WOOOORK FUCKKKK!!!!)
# from keras.utils.vis_utils import plot_model
# destination = './../../Desktop/'
# plot_model(model, to_file=destination + 'HANDWRITINGRECOGNITION.png', show_shapes=True, show_layer_names=True)
# img = mpimg.imread('./../../Desktop/HANDWRITINGRECOGNITION.png')
# plt.figure(figsize=(30, 15))
# imgplot = plt.imshow(img)

# cv2.imshow('HANDWRITING RECOGNITION', './../../Desktop/HANDWRITINGRECOGNITION.png')