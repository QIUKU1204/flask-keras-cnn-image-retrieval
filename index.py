# coding=utf-8
# comment=****
# author=QIUKU

import h5py
import numpy as np
import argparse
from extract_cnn_vgg16_keras import VGGNet
import os

# hide TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
''' hide FutureWarning:it will be treated as `np.float64 == np.dtype(float).type`.
    from ._conv import register_converters as _register_converters'''
# TODO

'''argument parse .
   get a dict of command line argument'''
ap = argparse.ArgumentParser()
ap.add_argument("-database", required=True,
                help="Path to database which contains images to be indexed")
ap.add_argument("-index", required=True,
                help="Name of index file")
args = vars(ap.parse_args())
print(args)


# Returns a list of filenames for all jpg images in a directory.
def get_imlist(path):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
   Extract features and index the images
'''
if __name__ == "__main__":

	db = args["database"]
	img_list = get_imlist(db)
	print(img_list)

	print("--------------------------------------------------")
	print("       feature extraction starts ")
	print("--------------------------------------------------")

	feats = []
	names = []

	model = VGGNet()
	for i, img_path in enumerate(img_list):
		norm_feat = model.extract_feature(img_path)
		'''os.path.split(PATH)函数以PATH的最后一个'\'作为分隔符，返回目录名和文件名组成的元组
		   索引0为目录名，索引1则为文件名'''
		img_name = os.path.split(img_path)[1]
		feats.append(norm_feat)
		names.append(img_name)
		print("extracting feature from image No. %d , %d images in total -> " % ((i + 1), len(img_list)) + img_name)

	print("--------------------------------------------------")
	print("      writing feature extraction results ...")
	print("--------------------------------------------------")

	# file of storing extracted features
	saved_filename = args["index"]
	# class File
	h5f = h5py.File(saved_filename, 'w')
	print(feats[8])
	print("--------------------------------np.array-------------------------------------------")
	feats = np.array(feats)
	print(feats)
	feats_ascii = []

	# create_dataset method only accepts ascii encoding character
	names_ascii = []
	for element in names:
		# encode: str -> bytes ; 编码: 字符串 -> 字节码
		names_ascii.append(str(element).encode("ascii"))
	h5f.create_dataset('dataset_1', data=feats)
	h5f.create_dataset('dataset_2', data=names_ascii)
	h5f.close()
	'''
	# test character encoding
	names_ascii_decode = []
	for element in names_ascii:
	    # decode: bytes -> str ; 解码: 字节码 -> 字符串
		names_ascii_decode.append(bytes(element).decode("utf-8", "ignore"))
	print(names)
	print(names_ascii_decode)
	'''

