from data import *
from os import listdir
from PIL import Image
import cv2
import random
import numpy as np

gray = False

data_image = "C://Users//HUE\Documents//Fiamma//byePlastic//Dataset//imageGRAY//" if gray else "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//imageRGB//"
data_label = "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//label//" if gray else "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//label//"
data_test = "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//Photos-001//" if gray else "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//Photos-001//"
data_image_512 = "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//png//imageGRAY//train//" if gray else "C://Users\HUE//Documents//Fiamma//byePlastic//Dataset//png//imageRGB//train//"
data_label_512 = "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//png//label//" if gray else "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//png//label//"
data_test_512 = "C://Users//HUE//Documents//Fiamma//byePlastic//Dataset//png//imageGRAY//test//" if gray else "C://Users\HUE//Documents//Fiamma//byePlastic//Dataset//png//imageRGB//test//"

data_image = data_test
data_image_512 = data_test_512

def cropImages (img, lbl, image_name, data_image_512, data_label_512) :
    w, h = img.size

    print(f"width: {w}; height: {h}")
    print("-----------------")

    offset_x = 0
    offset_y = 0

    while (offset_y + 512) <= h:

        while (offset_x + 512) <= w:
            image = img.crop((offset_x, offset_y, offset_x + 512, offset_y + 512))
            image.save(data_image_512 + str(image_name) + '.png', quality=100)
            print(f"Created image {image_name}")
            if (lbl):
                label = lbl.crop((offset_x, offset_y, offset_x + 512, offset_y + 512))
                label.save(data_label_512 + str(image_name) + '.png', quality=100)
                print(f"Created mask {image_name}")
            image_name += 1
            offset_x += 512
            
        else:
            image = img.crop(((w - 512), offset_y, w, offset_y + 512))
            image.save(data_image_512 + str(image_name) + '.png', quality=100)
            print(f"Created image {image_name}")
            if (lbl):
                label = lbl.crop(((w - 512), offset_y, w, offset_y + 512))
                label.save(data_label_512 + str(image_name) + '.png', quality=100)
                print(f"Created mask {image_name}")
            image_name += 1
            offset_x = 0
            offset_y += 512

    else:
        image = img.crop((offset_x, (h - 512), offset_x + 512, h))
        image.save(data_image_512 + str(image_name) + '.png', quality=100)
        print(f"Created image {image_name}")
        if (lbl):
            label = lbl.crop((offset_x, (h - 512), offset_x + 512, h))
            label.save(data_label_512 + str(image_name) + '.png', quality=100)
            print(f"Created mask {image_name}")
        image_name += 1
        offset_y += 512
    
    return image_name

def augmentationImages (image_name, data_image_512, data_label_512) :
    
    img = cv2.imread(data_image_512 + str(image_name - 1) + '.png')
    if (data_label_512):
        lbl = cv2.imread(data_label_512 + str(image_name - 1) + '.png')
    
    #Rotations each 45º (includes horizontal and vertical flips)
    angle = 45
    while angle < 360:
        rotation(img, angle, data_image_512, image_name)
        print(f"Created image {image_name}")
        if (data_label_512):
            rotation(lbl, angle, data_label_512, image_name)
            print(f"Created mask {image_name}")
        angle += 45
        image_name += 1
        
    
    # Channel Shift
    # channelShift(img, 60, data_image_512, image_name)
    # print(f"Created image {image_name}")
    # cv2.imwrite(data_label_512 + str(image_name) + '.png', lbl)
    # print(f"Created mask {image_name}")
    # image_name += 1
    
    return image_name


def rotation(img, angle, folder, image_name):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    imgRotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(folder + str(image_name) + '.png', imgRotated)


def channelShift(img, value, folder, image_name):
    value = int(random.uniform(-value, value))
    chShifted = img + value
    chShifted[:,:,:][chShifted[:,:,:]>255]  = 255
    chShifted[:,:,:][chShifted[:,:,:]<0]  = 0
    chShifted = chShifted.astype(np.uint8)
    cv2.imwrite(folder + str(image_name) + '.png', chShifted)

image_name = 0
for file in os.listdir(data_image):
    
    img = Image.open(data_image + file)
    
#     FOR TRAINING GRAY IMAGES
    if (gray):
        img = img.convert('L')
    
    # lbl = Image.open(data_label + file)
    
    print("-----------------")
    print(f"Image {file}")
    # image_name = cropImages(img, lbl, image_name, data_image_512, data_label_512)
    image_name = cropImages(img, None, image_name, data_image_512, None)
    
    # image_name = augmentationImages(image_name, data_image_512, data_label_512)
