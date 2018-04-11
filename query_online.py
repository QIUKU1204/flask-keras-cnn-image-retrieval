# coding=utf-8
# comment=****
# author=QIUKU
from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import os
# matplotlib module -- 图形可视化模块
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# hide TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
''' hide FutureWarning:it will be treated as `np.float64 == np.dtype(float).type`.
    from ._conv import register_converters as _register_converters'''
# TODO

'''argument parse .
   and get a dict of command line argument'''
ap = argparse.ArgumentParser()
ap.add_argument("-query", required=True,
                help="Path to query which contains image to be queried")
ap.add_argument("-index", required=True,
                help="Path to index")
ap.add_argument("-result", required=True,
                help="Path for output retrieved images")
args = vars(ap.parse_args())

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
img_feats = h5f['dataset_1'][:]
img_names = h5f['dataset_2'][:]
# decode: bytes -> str
img_names_decode = []
for element in img_names:
	img_names_decode.append(bytes(element).decode("utf-8", "ignore"))
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()
plt.close()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute similarity score and sort
query_img_feat = model.extract_feature(queryDir)
scores = np.dot(query_img_feat, img_feats.T)
# argsort函数返回按数组值从小到大的索引的数组(index_array)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
print("similarity score array: ", scores)
print("index_array: ", rank_ID)
print("sorted similarity score array: ", rank_score)

# number of retrieved images to show
max_ret = 3
ret_img_list = [img_names_decode[index] for i, index in enumerate(rank_ID[0:max_ret])]
print("retrieved %d images in order are: " % max_ret, ret_img_list)

# show top max_ret retrieved result one by one
for i, im in enumerate(ret_img_list):
	img_path = args["result"] + "/" + str(im)
	image = mpimg.imread(img_path)
	plt.title("search output %d" % (i + 1))
	plt.imshow(image)
	plt.show()
