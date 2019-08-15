

import os
import numpy as np
from PIL import Image
import cv2
from collections import Counter
import pickle as cPickle
import random, math
from data_util.bucketdata import BucketData


class AiTestDataGen(object):

    GO = 1
    EOS = 2

    def __init__(self,mean,
                 valid_target_len = float('inf'),
                 img_width_range = (12, 320),
                 word_len = 30,
                 channel=3):

        img_height = 32

        # '[(16,32),(27,32),(35,32),(64,32),(80,32)]'
        self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                             (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                             (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]


        self.bucket_min_width, self.bucket_max_width = img_width_range   #(12,320)
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.mean = mean
        self.channel = channel

        assert len(self.mean) == self.channel

        #320个bucket,每個放相應大小爲width圖像
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}


    def gen(self, cv_img):

        valid_target_len = self.valid_target_len
        img_bw, word = self.read_data(cv_img)
        if valid_target_len < float('inf'):
            word = word[:valid_target_len + 1]
        width = img_bw.shape[-1]

        # TODO:resize if > 320
        b_idx = min(width, self.bucket_max_width)

        self.bucket_data[b_idx].append(img_bw, word, 'test.jpg')

        b = self.bucket_data[b_idx].flush_out(self.bucket_specs,
                valid_target_length=valid_target_len,
                go_shift=1)
        if b is not None:
            return b
        else:
            assert False, 'no valid bucket of width %d'%width

        #self.clear()

    def read_data(self, cv_img):

        if len(cv_img.shape)==2:  # Gray iamge
            img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB))
        else:
            img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))
        w, h = img.size
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < float(self.bucket_min_width) / self.image_height:  # img_width_range[0]
            img = img.resize(
                (self.bucket_min_width, self.image_height),Image.ANTIALIAS)
        elif aspect_ratio > float(
                self.bucket_max_width) / self.image_height:    # img_width_range[1]
            img = img.resize(
                (self.bucket_max_width, self.image_height),Image.ANTIALIAS)
        elif h != self.image_height:
            img = img.resize(
                (int(aspect_ratio * self.image_height), self.image_height),
                Image.ANTIALIAS)

        # 以上操作將img圖像的寬resize爲12至320之間,高始終爲32
        if self.channel==1:
            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.float)
            img_bw = img_bw[np.newaxis, :]    # =>(1,h,w)
            img_bw[0] = (img_bw[0]-self.mean[0])/255.0
        elif self.channel==3:
            img_bw = np.asarray(img, dtype=np.float)
            img_bw = img_bw.transpose(2,0,1)  # =>(c,h,w)
            img_bw[0] = (img_bw[0]-self.mean[0])/255.0   # R
            img_bw[1] = (img_bw[1]-self.mean[1])/255.0   # G
            img_bw[2] = (img_bw[2]-self.mean[2])/255.0   # B
        else:
            raise ValueError

        word = np.array([self.GO,self.EOS], dtype=np.int32)

        return img_bw, word


def test_gen():

    print('testing DateGen...')

    s_gen = TestDataGen('../../data/date', 'train.txt','lexicon.txt')
    count = 0
    for batch in s_gen.gen(4):
        count += 1
        decoder_inputs = batch['decoder_inputs']
        grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in decoder_inputs]).transpose()]
        print(batch['filenames'])
        assert batch['data'].shape[2] == 32
    print(count)


if __name__ == '__main__':

    test_gen()
