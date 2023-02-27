'''
This code is to extract LRP feature from our pretrained regression models that are
trained on facial traits data;
find LRP feature by given a fixed output neuron;
experiment shows that 'lrp.sequential_preset_a_flat' can give a more clear illustration.

'''

# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:


from __future__ import\
    absolute_import, print_function, division, unicode_literals
# from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import innvestigate
import innvestigate.utils
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import preprocess_input
import scipy.io as sio
import gc

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))


if __name__ == "__main__":
    # load images, use a loosely cropped face data
    ro = 'images/'
    src_path = ro + 'faces_cropped_larger30_shiftup15/'
    lrp_tag = 'lrp.sequential_preset_a_flat'

    # choose which fold to visualize
    # foldn = 'fold4'
    dst_path = ro + 'loose_crop/' # + foldn + '/'

    # load pretrained models
    model_path = 'models/loose_crop/' # + foldn + '/'
    model_list = ['ASD_recognize_epoch20_batch4.h5',
                  'NT_recognize_epoch400_batch4.h5'
                  ]
    model_flag = ['ASD_recognize', 'NT_recognize']

    # set models
    for i in range(len(model_flag)): #len(model_flag)
        # if i >= 1: continue
        flag = model_flag[i]
        model_file = model_path + model_list[i]
        print(flag + '  :  ' + model_file)

        # Get model: redefined model using keras api, and set its weight by loading
        # pretrained model that is using tensorflow.keras
        base_model = VGG16(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_file)

        # V3: use neuron No. 0 of dense_2 output layer
        dst_raw_fig = dst_path + '1_raw_fig/' + flag + '/'
        if os.path.exists(dst_raw_fig) is False:
            os.makedirs(dst_raw_fig)

        dst_raw_data = dst_path + '1_raw_data_2d/' + flag + '/'


        # analyze
        analyzer = innvestigate.create_analyzer(
            lrp_tag, model,
            neuron_selection_mode="index"
        )

        # load face images
        fold_list = os.listdir(src_path)
        for fol_name in fold_list: # len(fol_list)
            print(fol_name)

            dst_raw_fig_fol = dst_raw_fig + fol_name
            if os.path.exists(dst_raw_fig_fol) is False:
                os.makedirs(dst_raw_fig_fol)
            dst_raw_data_fol = dst_raw_data + fol_name
            if os.path.exists(dst_raw_data_fol) is False:
                os.makedirs(dst_raw_data_fol)

            img_list = os.listdir(src_path + fol_name)
            for img_name in img_list:
                fi = src_path + fol_name + '/' + img_name
                image = utils.load_image(fi, 224)
                x = preprocess_input(image[None])

                # Apply analyzer w.r.t. maximum activated output-neuron
                result = analyzer.analyze(x, neuron_selection=0)
                # Aggregate along color channels and normalize to [-1, 1]
                a = result.sum(axis=np.argmax(np.asarray(result.shape) == 3))
                a /= np.max(np.abs(a))

                dict2 = {}
                dict2['data'] = -a[0]
                sio.savemat(dst_raw_data_fol + "/" + img_name.split('.')[0] + '.npy', dict2)

                # # Plot
                plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
                plt.axis('off')
                plt.savefig(dst_raw_fig_fol + "/" + img_name.split('.')[0] + '.jpg', bbox_inches='tight')
                plt.close()
                gc.collect()

