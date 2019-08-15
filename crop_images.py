import numpy as np
import os
import json
import cv2
import math
import codecs
import shutil
import glob

def getBbox(pts):

    pts = np.array(pts, dtype=np.int)
    rotate_rect = cv2.minAreaRect(pts)
    pts = cv2.boxPoints(rotate_rect)
    rect = cv2.boundingRect(pts)

    return rect[0],rect[1],rect[2]+rect[0],rect[3]+rect[1]

def getBBox(polygon):
    width = math.hypot(polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1])
    height = math.hypot(polygon[1][0] - polygon[2][0], polygon[1][1] - polygon[2][1])
    x = (polygon[0][0] + polygon[1][0] + polygon[2][0] + polygon[3][0]) / 4.0
    y = (polygon[0][1] + polygon[1][1] + polygon[2][1] + polygon[3][1]) / 4.0

    angle = math.atan2((polygon[1][1] - polygon[0][1]), (polygon[1][0] - polygon[0][0]))

    gtbox = ((x, y), (width, height), angle * 180 / 3.14)
    gtbox = cv2.boxPoints(gtbox)
    gtbox = np.array(gtbox, dtype="int")
    rect_gt = cv2.boundingRect(gtbox)  # 外接矩形

    rect_gt = [rect_gt[0], rect_gt[1], rect_gt[2], rect_gt[3]]

    rect_gt[2] += rect_gt[0]  # Convert GT width to right-coordinate
    rect_gt[3] += rect_gt[1]  # Convert GT height to bottom-coordinate

    return rect_gt



def classify_images():

    root = 'D:/workspace/data/AI_Hack_OCR'

    images_dir = os.path.join(root,'train_images')

    chinese_dir = os.path.join(images_dir,'chinese')
    if not os.path.exists(chinese_dir):
        os.makedirs(chinese_dir)

    english_dir = os.path.join(images_dir,'english')
    if not os.path.exists(english_dir):
        os.makedirs(english_dir)

    train_json = os.path.join(root, 'train_annotation.json')

    with open(train_json, "r", encoding='UTF-8') as f:
        json_root = json.load(f)

    for imgAnn in json_root['imgAnns']:
        img_name = imgAnn['file_name']
        if imgAnn['language'] == 'chinese':
            shutil.move(os.path.join(images_dir,img_name),os.path.join(chinese_dir,img_name))
        elif imgAnn['language'] == 'english':
            shutil.move(os.path.join(images_dir, img_name), os.path.join(english_dir, img_name))


def classify_test():

    root = 'D:/workspace/data/AI_Hack_OCR'

    images_dir = os.path.join(root, 'test_images')

    chinese_dir = os.path.join(images_dir, 'chinese')
    if not os.path.exists(chinese_dir):
        os.makedirs(chinese_dir)

    english_dir = os.path.join(images_dir, 'english')
    if not os.path.exists(english_dir):
        os.makedirs(english_dir)

    images_list = glob.glob(images_dir+'/*.jpg')


    for img_file in images_list:

        img_name = os.path.basename(img_file)

        img = cv2.imread(img_file)

        h,w = img.shape[:2]

        if h==2048 and w==2048:
            shutil.move(os.path.join(images_dir, img_name), os.path.join(chinese_dir, img_name))
        else :
            shutil.move(os.path.join(images_dir, img_name), os.path.join(english_dir, img_name))


if __name__ == '__main__':

    root = 'D:/workspace/data/AI_Hack_OCR'
    
    train_json = os.path.join(root,'train_annotation.json')
    
    chinese_label = codecs.open(os.path.join(root,'chinese','lables.txt'),'w','utf-8')
    
    english_label = codecs.open(os.path.join(root, 'english', 'lables.txt'), 'w','utf-8')
    
    with open(train_json, "r", encoding='UTF-8') as f:
        json_root = json.load(f)
    
    for imgAnn in json_root['imgAnns']:
    
        if not os.path.exists(os.path.join(root,'train_images',imgAnn['file_name'])):
            continue
    
        img = cv2.imread(os.path.join(root,'train_images',imgAnn['file_name']))
        print(imgAnn['file_name'])
    
        for i,obj in enumerate(imgAnn['annotations']):
    
            bbox = getBbox(obj['polygon'])
            i += 1
            # bbox = getBBox(obj['polygon'])
            # print(bbox)
    
            crop_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    
            dst_path = os.path.join(root,imgAnn['language'],'images')
    
            dst_file_name = imgAnn['file_name'].split('.')[0]+'_'+str(i).zfill(4)+'.jpg'
    
            cv2.imwrite(os.path.join(dst_path,dst_file_name),crop_img)
    
            if imgAnn['language'] == 'chinese':
                chinese_label.write(dst_file_name)
                chinese_label.write('\t')
                chinese_label.write(obj['text'])
                chinese_label.write('\n')
    
            elif imgAnn['language'] == 'english':
    
                english_label.write(dst_file_name)
                english_label.write('\t')
    
                english_label.write(obj['text'])
    
                english_label.write('\n')
    
    chinese_label.close()
    
    english_label.close()

