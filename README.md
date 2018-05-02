# Image Retrieval Engine Based on Keras

## 环境

```
In [1]: import keras
Using TensorFlow backend.
```

keras 2.1.5 经过测试可用;
修改后已支持Python3.6环境;
此外需要numpy, matplotlib, os, h5py, argparse;
推荐使用anaconda+pycharm的开发组合。

### 使用

- 步骤一

`python index.py -database <path-to-dataset> -index <name-for-output-index>`

- 步骤二

`python query_online.py -query <path-to-query-image> -index <path-to-index-flie> -result <path-to-images-for-retrieval>`

```sh
├── database 图像数据集
├── extract_cnn_vgg16_keras.py 使用预训练vgg16模型提取图像特征
|── creat_index.py 对图像集提取特征，建立索引
├── query_online.py 基于web的搜索
├── image_query.py 库内搜索
└── README.md
```

#### 示例

```sh
# 对database_less文件夹内图片进行特征提取，建立索引文件CNN_extracted_image_feature.h5
python index.py -database database-few -index CNN_extracted_image_feature_few.h5

# 将1-001.jpg作为Query图片，在database_less内使用CNN_extracted_image_feature.h5进行相似图片查找，并显示最相似的3张图片
python query_local.py -query database-few/1-001.jpg -index CNN_extracted_image_feature_few.h5 -result database-few
```

### 更新

- keras更新到2.1.5版本，并且特征提取代码大幅精简；
- 显示检索得到的图片，可自由定义查询图片及检索图片集；
- 2018-04-11: 修改了Python2.7环境下的原项目，使之支持Python3.6环境下的运行;
- 2018-04-13: 解决了字符编码的问题，使之能够支持UTF-8编码方式;

### 目标

重新用flask写CNN-Web-Demo-for-Image-Retrieval，使它支持在线上传功能。

### 论文推荐

[**awesome-cbir-papers**](https://github.com/willard-yuan/awesome-cbir-papers)
