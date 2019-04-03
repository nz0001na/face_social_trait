import shutil
import os

des_path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Dani_141124_2_shorten_rename/'
if os.path.exists(des_path) is False:
    os.makedirs(des_path)

path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Dani_141124_2_shorten_raw/'
list = os.listdir(path)
for j in range(len(list)):
    name = list[j]
    names = name.split('.')[0].split('_')
    new_name = 'frame_' + format(j+1,'06d') + '.png'
    shutil.copy(path + name, des_path + new_name)
    # os.rename('a.txt', 'b.klm')

print 'done'


