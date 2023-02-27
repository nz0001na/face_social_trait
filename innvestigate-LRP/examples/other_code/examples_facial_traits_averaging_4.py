'''
This code is to load average face to display
'''

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import gc

if __name__ == "__main__":
    # load images
    ro = 'images/'
    lrp_list = ['lrp.sequential_preset_a_flat'
                # 'lrp.sequential_preset_b_flat',
                # "lrp.epsilon", "lrp.z"
                ]
    model_flag = ['ASD_strong', 'NT_strong', 'ASD_warm', 'NT_warm']
    src_norm_path = ro + 'V3_neuron_all_npy_2d_flip_avg_face/'

    # set models
    for i in range(len(model_flag)): # len(model_flag)
        flag = model_flag[i]
        print(flag)

        for q in range(len(lrp_list)):
            lrp_tag = lrp_list[q]
            print('    ' + lrp_tag)

            src_item_norm_path = src_norm_path + lrp_tag + '/' + flag + '/'

            dd = sio.loadmat(src_item_norm_path + lrp_tag + '_' + flag + 'all.mat')
            data = dd['data']
            flatten_data = []
            for m in range(len(data)):
                flatten_data.append(data[m])

            sum_dat = np.asarray(flatten_data).sum(axis=0)
            avg_data = sum_dat / 500.0
            gg = {}
            gg['data'] = avg_data
            sio.savemat(src_item_norm_path + lrp_tag + '_' + flag + '_avg_9.mat', gg)
            final = avg_data.reshape(224, 224)
            # # Plot
            plt.imshow(final, cmap="seismic", clim=(-0.1, 0.1))  # ,
            plt.axis('off')
            plt.savefig(src_norm_path + lrp_tag + '_' + flag + '_avg_9.png')
            plt.close()
            gc.collect()



            # ab= np.asarray(data).sum(axis=0)
            # abb = ab/len(data)
            # a = sum(data) / len(data)
            # #
            # a = abb.sum(axis=np.argmax(np.asarray(abb.shape) == 3))
            # a /= np.max(np.abs(a))





