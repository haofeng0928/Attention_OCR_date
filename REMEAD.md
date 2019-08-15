### 训练

工程根目录提供了三个脚本，train_coco.sh，train_ai_hack_eng.sh, train_all_eng.sh分别用于训练coco中crop_words数据集、比赛中英文部分数据集及训练coco+比赛一起的数据集。

###### 注意

1. 修改对应脚本中data-path,data-root-dir及lexicon-file文件。字典文件lexicon-file可以使用data/eng_lexicon.txt文件。
2. 执行crop_images.py脚本，可将比赛中训练集图片中每个字符crop出来，作为训练的数据集方式
3. 执行coco-text.py脚本，可将原始的coco中crop-word的数据集转换为算法需要的训练的数据格式



### 测试

测试脚本是src/ai-hack-test.py文件，该文件会依赖使用east算法检测出的test_eng_det_res_v2.json，具体可根据自己的目录修改该脚本中以下几个目录。

``` bash
model_dir = '../models'
images_root = '/home/xiaguomiao/data/ai-hack/test/english/images'
src_json_file = '/home/xiaguomiao/data/ai-hack/test/english/test_eng_det_res_v2.json'
res_json_file = '/home/xiaguomiao/data/ai-hack/test/english/eng_submit.json'
lexicon_file = '/home/xiaguomiao/data/ai-hack/eng_lexicon.txt'
result_dir = '../results'
```

