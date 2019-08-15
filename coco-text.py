
import os
import codecs
from collections import defaultdict
import pprint


def convert_txt(src_file, dst_file):

    # root = 'D:/workspace/data/COCO-Text-words-trainval'
    root = 'D:/myProject/myData/COCO-Text-words-trainval'

    src = os.path.join(root, src_file)

    dst = os.path.join(root, dst_file)

    f = codecs.open(src, 'r', 'utf-8')

    lines = f.readlines()

    f.close()

    f = codecs.open(dst, 'w', 'utf-8')

    for line in lines:
        line = line.strip('')
        if len(line)<5:
            continue
        line = line.replace(',', '.jpg\t', 1)
        f.write(line)

    f.close()


def filter():

    root = 'D:/workspace/data/AI_Hack_OCR/english'

    f = codecs.open('lexicon.txt', 'r', 'utf-8')
    lines = f.readlines()
    f.close()

    lexicon = []
    for line in lines:
        line = line.strip()
        lexicon.append(line)

    f = codecs.open(os.path.join(root, 'lables.txt'), 'r', 'utf-8')
    lines = f.readlines()
    f.close()

    f = codecs.open(os.path.join(root, 'english_train.txt'), 'w', 'utf-8')

    for line in lines:
        line1 = line.strip()
        s = line1.split()

        flage = True
        if len(s)!= 2:
            continue

        if len(s[1])>=32:
            print(line1)
            continue

        for c in s[1]:
            if c not in lexicon:
                flage = False
                break

        if flage:
            f.write(s[0])
            f.write('\t')
            f.write(s[1])
            f.write('\n')

    f.close()

def ai_hack_filter():

    root = 'D:/workspace/data/AI_Hack_OCR/english'

    f = codecs.open(os.path.join(root, 'english_train.txt'), 'r', 'utf-8')

    lines = f.readlines()

    f.close()

    f = codecs.open(os.path.join(root, 'new_english_train.txt'), 'w', 'utf-8')

    for line in lines:
        line = line.strip()
        s = line.split()
        img_file = os.path.join(root,'images',s[0])
        if os.path.exists(img_file):
            f.write(s[0])
            f.write('\t')
            f.write(s[1])
            f.write('\n')

    f.close()


def generate_lexicon():

    root = 'D:/myProject/myData/COCO-Text-words-trainval'

    f = codecs.open(os.path.join(root, 'train.txt'), 'r', 'utf-8')
    lines = f.readlines()
    f.close()

    f = codecs.open(os.path.join(root, 'val.txt'), 'r', 'utf-8')
    lines += f.readlines()
    f.close()
    #
    # root = 'D:/workspace/data/AI_Hack_OCR/english'
    #
    # f = codecs.open(os.path.join(root, 'lables.txt'), 'r', 'utf-8')
    # lines += f.readlines()
    # f.close()
    #
    # lexicon = defaultdict(int)
    #
    # for line in lines:
    #     line = line.strip()
    #     label = line.split('\t')[-1]
    #     for c in label:
    #         lexicon[c] += 1
    #
    # pprint.pprint(lexicon)

    tem = []
    dst = os.path.join(root, 'labels.txt')
    f = codecs.open(dst, 'w', 'utf-8')
    for line in lines:
        line = line.strip().split()[-1]
        for c in line:
            if c not in tem:
                tem.append(c)
    tem = sorted(tem)
    for c in tem:
        f.write(c + '\n')
    # print(tem)
    f.close()


if __name__ == '__main__':
    # convert_txt('train_words_gt.txt', 'train.txt')
    # convert_txt('val_words_gt.txt', 'val.txt')
    generate_lexicon()

