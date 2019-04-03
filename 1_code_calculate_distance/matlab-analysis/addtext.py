from PIL import Image, ImageDraw, ImageFont
import os
import csv
from decimal import Decimal

des_path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_shorten_with_text_v3/'
if os.path.exists(des_path) is False:
    os.makedirs(des_path)

speed_path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_speed_v3/'

left_data = []
right_data = []
f = csv.reader(open(speed_path + 'left_speed.csv', 'rb'))
for row in f:
    left_data.append(row[0])

f = csv.reader(open(speed_path + 'right_speed.csv', 'rb'))
for row in f:
    right_data.append(row[0])

imgDir = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_shorten_rename/'
list = os.listdir(imgDir)
for i in range(1,len(list)):
    name = list[i]
    # create Image object with the input image
    image = Image.open(imgDir + name)

    # initialise the drawing context with
    # the image object as background
    draw = ImageDraw.Draw(image)
    # create font object with the font file and specify
    # desired size
    # font = ImageFont.truetype('Arial.ttf', size=45)
    # starting position of the message

    index = int((i-1) / 30)
    left = left_data[index]
    right = right_data[index]

    (x, y) = (20, 20)
    # message = 'left:'+ str(Decimal(left).quantize(Decimal('0.0000'))) + ' ; right:' + str(Decimal(right).quantize(Decimal('0.0000')))
    message = 'right:' + str(Decimal(right).quantize(Decimal('0.0000'))) + ' ; left:'+ str(Decimal(left).quantize(Decimal('0.0000')))
    color = 'rgb(0, 0, 0)'  # black color
    # draw the message on the background
    draw.text((x, y), message, fill=color)

    # save the edited image

    image.save(des_path + name)