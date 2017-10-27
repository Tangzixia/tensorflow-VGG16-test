#-*-coding=utf-8-*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image

dir="test_data_std/"

def getFileArr(dir):
	result_arr=[]
	label_list=[]
	map={}
	map_file_result={}
	map_file_label={}
	map_new={}
	count_label=0
	count=0

	file_list=os.listdir(dir)
	for file in file_list:
		file_path=os.path.join(dir,file)

		label=file.split(".")[0].split("_")[0]
		map[file]=label
		if label not in label_list:
			label_list.append(label)
			map_new[label]=count_label
			count_label=count_label+1
		img=Image.open(file_path)
		result=np.array([])
		r,g,b=img.split()

		r_arr=np.array(r).reshape(224*224)
		g_arr=np.array(g).reshape(224*224)
		b_arr=np.array(b).reshape(224*224)
		img_arr=np.concatenate((r_arr,g_arr,b_arr))
		result=np.concatenate((result,img_arr))
		#result=result.reshape((3,112,112))
		result=result.reshape((224,224,3))
		result=result/255.0
		map_file_result[file]=result
		result_arr.append(result)
		count=count+1
	for file in file_list:
		map_file_label[file]=map_new[map[file]]
		#map[file]=map_new[map[file]]
	
	ret_arr=[]
	for file in file_list:
		each_list=[]
		label_one_zero=np.zeros(count_label)
		result=map_file_result[file]
		label=map_file_label[file]
		#label_one_zero[label]=1.0
		#print(label_one_zero)
		each_list.append(result)
		each_list.append(label)
		ret_arr.append(each_list)
	np.save('test_data.npy', ret_arr)
	return ret_arr

def to_categorial(y,n_classes):
	y_std=np.zeros([len(y),n_classes])
	for i in range(len(y)):
		y_std[i,y[i]]=1.0
	return y_std

def load_data(train_dir="train_data.npy",
			test_dir="test_data.npy"):
	train_data=np.load(train_dir)
	test_data=np.load(test_dir)
	X_train_non,y_train_non=train_data[:,0],train_data[:,1]
	X_test_non,y_test_non=test_data[:,0],test_data[:,1]

	X_train=np.zeros([len(X_train_non),224,224,3])
	X_test=np.zeros([len(X_test_non),224,224,3])
	for i in range(len(X_train_non)):
		X_train[i,:,:,:]=X_train_non[i]
	for i in range(len(X_test_non)):
		X_test[i,:,:,:]=X_test_non[i]
	#y_train_non=y_train_non.tolist()
	#y_test_non=y_test_non.tolist()
	y_train=to_categorial(y_train_non,1000)
	y_test=to_categorial(y_test_non,1000)	
	return (X_train,y_train),(X_test,y_test)
