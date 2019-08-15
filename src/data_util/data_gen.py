__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
import traceback
from .bucketdata import BucketData


class DataGen(object):

    GO = 1
    EOS = 2

    def __init__(self,
                 data_root,
                 annotation_fn,
                 lexicon_file,
                 evaluate,
                 mean,
                 channel,
                 valid_target_len=float('inf'),

                 img_width_range=(12, 320),
                 word_len=30):
        """
        :param data_root:
        :param annotation_fn: annotation file name
        :param lexicon_file: lexicon file name
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 32
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.data_root = data_root
        self.mean = mean
        self.channel = channel
        assert len(self.mean) == self.channel

        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(self.data_root, annotation_fn)

        if not os.path.exists(lexicon_file):
            lexicon_file = os.path.join(self.data_root, lexicon_file)

        if evaluate:
            '[(16,32),(27,32),(35,32),(64,32),(80,32)]'
            # self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)),
            #                      (int(math.floor(108 / 4)), int(word_len + 2)),
            #                      (int(math.floor(140 / 4)), int(word_len + 2)),
            #                      (int(math.floor(256 / 4)), int(word_len + 2)),
            #                      (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
            self.bucket_specs = [(int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            '[(16,11),(27,17),(35,19),(64,22),(80,32)]'
            # self.bucket_specs = [(int(64 / 4), 9 + 2),
            #                      (int(108 / 4), 15 + 2),
            #                      (int(140 / 4), 17 + 2),
            #                      (int(256 / 4), 20 + 2),
            #                      (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]
            self.bucket_specs = [(int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range  # (12, 320)
        self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}

        self.lexicon_dic = {}
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                self.lexicon_dic[line] = i+3

    def clear(self):
        self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                s = l.strip('\n').split('\t')
                if len(s) == 1:
                    s = l.strip('\n').split(' ')
                if len(s) != 2:
                    print(l.strip('\n'))
                    continue

                img_path, lex = s

                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if img_bw is None or word is None:
                        continue
                    
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]

                    # TODO:resize if width > 320
                    width = img_bw.shape[-1]
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root, img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(self.bucket_specs,
                                                              valid_target_length=valid_target_len, go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d' % width
                # ignore error images
                # except IOError:
                #     print('IOError!')
                #     pass
                except Exception:
                    print('Error!')
                    traceback.print_exc()
        print('gen ends.')
        self.clear()

    def read_data(self, img_path, lex):
        """
        resize the img and add start and end flag for lex
        :param img_path:
        :param lex:  label
        :return:
        """
        if len(lex) == 0 or len(lex) >= self.bucket_specs[-1][1]:
            return None, None
        
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            if w < 10 and h < 10:
                return None, None
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:  # 12 / 32 = 0.375
                img = img.resize((self.bucket_min_width, self.image_height), Image.ANTIALIAS)
            elif aspect_ratio > float(self.bucket_max_width) / self.image_height:  # 320 / 32 = 10
                img = img.resize((self.bucket_max_width, self.image_height), Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize((int(aspect_ratio * self.image_height), self.image_height), Image.ANTIALIAS)

            if self.channel == 1:
                img_bw = img.convert('L')  # gray
                img_bw = np.asarray(img_bw, dtype=np.float)
                img_bw = img_bw[np.newaxis, :]  # (1,h,w)
                img_bw[0] = (img_bw[0]-self.mean[0])/255.0
            elif self.channel == 3:
                img_bw = np.asarray(img, dtype=np.float)
                img_bw = img_bw[:, :, ::-1]  # RGB to BGR
                img_bw = img_bw.transpose(2, 0, 1)  # (3,h,w)
                img_bw[0] = (img_bw[0]-self.mean[0])/255.0  # B
                img_bw[1] = (img_bw[1]-self.mean[1])/255.0  # G
                img_bw[2] = (img_bw[2]-self.mean[2])/255.0  # R
            else:
                raise ValueError

        word = [self.GO]
        lex = lex.lower()
        for c in lex:
            if c in self.lexicon_dic.keys():
                word.append(self.lexicon_dic[c])
        word.append(self.EOS)
        
        if len(word) == 2:
            return img_bw, None

        word = np.array(word, dtype=np.int32)

        return img_bw, word


def test_gen():

    print('testing DateGen...')

    s_gen = DataGen('../../data/date', 'train.txt','lexicon.txt')
    count = 0
    for batch in s_gen.gen(4):
        count += 1
        decoder_inputs = batch['decoder_inputs']
        grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in decoder_inputs]).transpose()]
        print(batch['filenames'])
        assert batch['data'].shape[2] == 32
    print(count)


