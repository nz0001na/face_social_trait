'''
this code is to fine tune a pretrained model
a regression model
code is from: https://keras.io/api/applications/
'''

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import scipy.io as sio
import random
import numpy as np
import os
import math
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# set ASD or NT
# flag = 'ASD'
flag = 'NT'
# set facial traits: warm, critical, competent, practical, feminine, strong
#                    youthful, charismatic, trustworthy,  dominant, recognize
trait = 'recognize'
batchsize = 4
epochnum = 450
nn_name = 'vgg16'
fold_num = '3'   # 1,2,3,4,5

print(nn_name + ' : ' + flag + ' : ' + trait + ' : fold-' + fold_num)
ro = '/home/na/4_ASD_facial_traits/3_data/'
img_path = ro + '4_CelebA_face_new_cropped_category/faces_cropped_larger30_shiftup15/'
dst_path = ro + '5_experiment/finetuned_saved_models_larger30_shiftup15/fold' + fold_num + '/' + flag + '/' + trait + '/' + nn_name + '/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

# read train and test list
print('** Reading train and test list....')
fold_ids = {}
list_path = ro + '5_experiment/5-fold_list/'
list_file = os.listdir(list_path)
for p in range(len(list_file)):
    file_name = list_file[p]
    ids = []
    f = csv.reader(open(list_path + file_name, 'r', newline=''))
    for row in f:
        ids.append(row[0])
    fold_ids[str(p+1)] = ids

test_id = []
train_id = []
for m in fold_ids:
    if m == fold_num:
        test_id.append(fold_ids[m])
    else:
        train_id.append(fold_ids[m])

test_id = [item for sublist in test_id for item in sublist]
train_id = [item for sublist in train_id for item in sublist]

### load data
print('** Reading data....')
data_file = ro + '4_CelebA_face_new_cropped_category/rate' + flag + '/' + trait + '.mat'
data = sio.loadmat(data_file)

train_img_list = []
train_label_list = []
test_img_list = []
test_label_list = []
for p in range(len(train_id)):
    id = train_id[p]
    train_label_list.append(data[id])
    li = os.listdir(img_path + id + '/')
    li.sort()
    train_img_list.append(li)

for p in range(len(test_id)):
    id = test_id[p]
    test_label_list.append(data[id])
    li = os.listdir(img_path + id + '/')
    li.sort()
    test_img_list.append(li)

print('**** generating training data....')
trainImagesX = []
trainY = []
testImagesX = []
testY = []
for q in range(len(train_img_list)):
    img_folder = train_img_list[q]
    for k in range(len(img_folder)):
        img_name = img_folder[k]
        id = train_id[q]
        fi = img_path + id + '/' + img_name
        img = image.load_img(fi, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.asarray(x)
        trainImagesX.append(x)
        trainY.append(train_label_list[q][0][k])

for q in range(len(test_img_list)):
    img_folder = test_img_list[q]
    for k in range(len(img_folder)):
        img_name = img_folder[k]
        id = test_id[q]
        fi = img_path + id + '/' + img_name
        img = image.load_img(fi, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.asarray(x)
        testImagesX.append(x)
        testY.append([str(test_label_list[q][0][k])])

# with open(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize)+'_true_label.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows(testY)

# preprocessing data
trainImagesX = np.asarray(trainImagesX)
trainImagesX = np.reshape(trainImagesX, (400, 224,224,3)).astype('float32')/255
trainY = np.asarray(trainY)
trainY = trainY.astype('float32')

testImagesX = np.asarray(testImagesX)
testImagesX = np.reshape(testImagesX, (100, 224,224,3)).astype('float32')/255
testY = np.asarray(testY)
testY = testY.astype('float32')


### create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='linear')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# tf.keras.metrics.RootMeanSquaredError()



# train the model on the new data for a few epochs
# train the model
print("training model...")
# tb_callback = tf.keras.callbacks.TensorBoard('/home/zn/logs', update_freq=1)
# model.fit(x_train, y_train, callbacks=[tb_callback])
# validation_data=(valImagesX, valY),
history = model.fit(x=trainImagesX, y=trainY,  epochs=epochnum, batch_size=batchsize)




loss_train = history.history['loss']
# loss_val = history.history['val_loss']
rmse_train = history.history['root_mean_squared_error']
# rmse_val = history.history['val_root_mean_squared_error']
epochs = range(0, epochnum)
plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.plot(epochs, rmse_train, 'r', label='Training RMSE')
# plt.plot(epochs, rmse_val, 'tab:orange', label='validation RMSE')
plt.title('Training loss/RMSE')
plt.xlabel('Epochs')
plt.ylabel('Loss/RMSE')
plt.legend()
plt.savefig(dst_path + 'loss_RMSE_epoch'+str(epochnum)+'_batch'+str(batchsize) + '.jpg')
# plt.show()

#  save models
# model.save(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize))
# loaded_model = tf.keras.models.load_model(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize))
model.save(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize)+'.h5')
loaded_model2 = load_model(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize)+'.h5')


# predict test data
y_pred = model.predict(testImagesX)
y_pred_sav2 = loaded_model2.predict(testImagesX)
print(flag + ' : ' + trait + ' : ' + str(fold_num) + ' ***** Save results.....')
print(str(epochnum) + ' : ' + str(batchsize))
with open(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize)+'_predicted_label.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(y_pred)
with open(dst_path + 'epoch'+str(epochnum)+'_batch'+str(batchsize)+'_predicted_label_sav2.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(y_pred_sav2)

# # compute R
print('**** result: trained model')
rmse = mean_squared_error(testY, y_pred, squared=False)
print('    RMSE = ' + str(rmse))
r2 = r2_score(testY, y_pred)
print('    R2 =' + str(r2))
corr, p_value = pearsonr(testY.flatten(), y_pred.flatten())
print('    Pearsons correlation:' + str(corr))

# # compute R
# print('**** result: saved model')
# rmse = mean_squared_error(testY, y_pred_sav, squared=False)
# print('    RMSE = ' + str(rmse))
# r2 = r2_score(testY, y_pred_sav)
# print('    R2 =' + str(r2))
# corr, p_value = pearsonr(testY.flatten(), y_pred_sav.flatten())
# print('    Pearsons correlation:' + str(corr))

# # compute R
print('**** result: saved model h5')
rmse = mean_squared_error(testY, y_pred_sav2, squared=False)
print('    RMSE = ' + str(rmse))
r2 = r2_score(testY, y_pred_sav2)
print('    R2 =' + str(r2))
corr, p_value = pearsonr(testY.flatten(), y_pred_sav2.flatten())
print('    Pearsons correlation:' + str(corr))

