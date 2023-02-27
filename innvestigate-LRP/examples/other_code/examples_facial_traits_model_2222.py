'''
This code is to extract LRP feature from our pretrained models which are
trained on facial traits data
'''

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

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "../utils.py"))

if __name__ == "__main__":
    # load images
    ro = 'images/'
    data_list = ['faces_fold1_cropped']  # , 'faces_fold1_raw'

    lrp_list = ["lrp.epsilon", "lrp.z",
                "lrp.w_square", "lrp.flat", "lrp.alpha_1_beta_0", 'lrp.z_plus',
                'lrp.sequential_preset_a', 'lrp.sequential_preset_b',
                'lrp.sequential_preset_a_flat', 'lrp.sequential_preset_b_flat',
                'lrp.sequential_preset_b_flat_until_idx'
                ]
    # "lrp.z_IB", "lrp.epsilon_IB", "lrp.alpha_2_beta_1",
    # "lrp.alpha_2_beta_1_IB", "lrp.alpha_1_beta_0_IB", 'lrp.z_plus_fast',

    model_path = '../models/'
    model_list = ['ASD_strong_epoch200_batch4.h5',
                  'NT_strong_epoch200_batch4.h5',
                  'ASD_warm_epoch300_batch4.h5',
                  'NT_warm_epoch200_batch4.h5'
                  ]
    model_flag = ['ASD_strong', 'NT_strong', 'ASD_warm', 'NT_warm']

    # set models
    for i in range(len(model_flag)): #len(model_flag)
        flag = model_flag[i]
        model_file = model_path + model_list[i]
        print(flag)
        print('    ' + model_file)

        # Get model
        base_model = VGG16(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.load_weights(model_file)
        # new_model = Model(inputs=model.inputs, outputs=model.get_layer('dense_1').output)

        for q in range(0, 2):  # len(lrp_list)
            lrp_tag = lrp_list[q]
            print('    ' + lrp_tag)
            # V1: use whole model
            # V2: use dense_1 layer
            # V3: use dense_2 output layer with neuron No. 0
            # V4: use dense_2 output layer with neuron No. 0 + Normalized faces
            dst_path_model = ro + 'V4_neuron/' + lrp_tag + '/' + flag + '/'
            if os.path.exists(dst_path_model) is False:
                os.makedirs(dst_path_model)

            analyzer = innvestigate.create_analyzer(
                lrp_tag, model,
                neuron_selection_mode="index"
            )
            # load face images
            for m in data_list:
                fol_list = os.listdir(ro + m)
                dst_fol = dst_path_model + m
                if os.path.exists(dst_fol) is False:
                    os.makedirs(dst_fol)

                for n in range(0, len(fol_list)): # len(fol_list)
                    fol_name = fol_list[n]
                    dst_fol2 = dst_path_model + m + '/' + fol_name
                    if os.path.exists(dst_fol2) is False:
                        os.makedirs(dst_fol2)

                    img_list = os.listdir(ro + m + '/' + fol_name)
                    for p in range(len(img_list)):
                        img_name = img_list[p]
                        fi = ro + m + '/' + fol_name + '/' + img_name
                        img = utils.load_image(fi, 224)

                        # image1 = np.array(cv2.imread('images/zn/020607.jpg'))
                        # resized = cv2.resize(image1, (224, 224), interpolation=cv2.INTER_AREA)
                        # img2 = resized[..., ::-1] / 255.0
                        img2 = (img / 255.).astype(np.float32)
                        # #  normalized to be given as input to the VGG - 16 network.
                        mean = np.array([0.3825, 0.4576, 0.6227]).astype(np.float32)
                        std = np.array([0.2029, 0.2117, 0.2475]).astype(np.float32)
                        imgX = (img2 - mean) / std
                        # image[..., 0] -= mean[0]
                        # image[..., 1] -= mean[1]
                        # image[..., 2] -= mean[2]
                        # image[..., 0] /= std[0]
                        # image[..., 1] /= std[1]
                        # image[..., 2] /= std[2]

                        # imgX0 = (image[np.newaxis] - mean) /std
                        # Add batch axis and preprocess
                        # imgX2 = preprocess_input(imgX[None])
                        x = imgX[np.newaxis]

                        # Add batch axis and preprocess
                        # x = preprocess_input(image[None])

                        # Apply analyzer w.r.t. maximum activated output-neuron
                        # a = analyzer.analyze(x)
                        a = analyzer.analyze(x, neuron_selection=0)  # , neuron_selection=5

                        # Aggregate along color channels and normalize to [-1, 1]
                        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
                        a /= np.max(np.abs(a))
                        # Plot
                        plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
                        plt.axis('off')
                        plt.savefig(dst_fol2 + "/" + img_name.split('.')[0] + '.png')


##########################################################################
        # Create analyzer: map.keys()
        # analyzers = {
        #     # Utility.
        #     "input": Input,
        #     "random": Random,
        #
        #     # Gradient based
        #     "gradient": Gradient,
        #     "gradient.baseline": BaselineGradient,
        #     "input_t_gradient": InputTimesGradient,
        #     "deconvnet": Deconvnet,
        #     "guided_backprop": GuidedBackprop,
        #     "integrated_gradients": IntegratedGradients,
        #     "smoothgrad": SmoothGrad,
        #
        #     # Relevance based
        #     "lrp": LRP,
        #     "lrp.z": LRPZ,
        #     "lrp.z_IB": LRPZIgnoreBias,
        #
        #     "lrp.epsilon": LRPEpsilon,
        #     "lrp.epsilon_IB": LRPEpsilonIgnoreBias,
        #
        #     "lrp.w_square": LRPWSquare,
        #     "lrp.flat": LRPFlat,
        #
        #     "lrp.alpha_beta": LRPAlphaBeta,
        #
        #     "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
        #     "lrp.alpha_2_beta_1_IB": LRPAlpha2Beta1IgnoreBias,
        #     "lrp.alpha_1_beta_0": LRPAlpha1Beta0,
        #     "lrp.alpha_1_beta_0_IB": LRPAlpha1Beta0IgnoreBias,
        #     "lrp.z_plus": LRPZPlus,
        #     "lrp.z_plus_fast": LRPZPlusFast,
        #
        #     "lrp.sequential_preset_a": LRPSequentialPresetA,
        #     "lrp.sequential_preset_b": LRPSequentialPresetB,
        #     "lrp.sequential_preset_a_flat": LRPSequentialPresetAFlat,
        #     "lrp.sequential_preset_b_flat": LRPSequentialPresetBFlat,
        #     "lrp.sequential_preset_b_flat_until_idx": LRPSequentialPresetBFlatUntilIdx,
        #
        #
        #     # Deep Taylor
        #     "deep_taylor": DeepTaylor,
        #     "deep_taylor.bounded": BoundedDeepTaylor,
        #
        #     # DeepLIFT
        #     #"deep_lift": DeepLIFT,
        #     "deep_lift.wrapper": DeepLIFTWrapper,
        #
        #     # Pattern based
        #     "pattern.net": PatternNet,
        #     "pattern.attribution": PatternAttribution,
        # }