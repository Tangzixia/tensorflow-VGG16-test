#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from read_files import *
import tensorflow as tf
from model import *

def min(x,y):
	if x<y:
		return x
	else:
		return y
def train_VGG(epoch,mini_batch_size,mini_batch):
	
	#batch_[]=[]
	#batch
	with tf.Session() as sess:
		sess.run(init_op)
		for i in range(epoch):
			for mini_batch in mini_batches:
				print("第0次迭代".format(i))
				sess.run(loss,feed_dict={x:mini_batch[0],y:mini_batch[1]})

	print(loss)
if __name__=="__main__":
	(x_train,y_train),(x_test,y_test)=load_data()
	x=tf.placeholder(tf.float32,[None,224,224,3])
	y_=tf.placeholder(tf.float32,[None,1000])
	y,logits=model(x,logits=True)
	
	#x=tf.reshape(x,[-1,224,224,3])
	cross_entropy=-tf.reduce_sum(y_*tf.log(y))
	train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

	init_op=tf.global_variables_initializer()
	mini_batch_size=200
	batch_size=int(np.ceil(len(x_train)/mini_batch_size))

	pred=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(pred,tf.float32))
	#mini_batches=(x_train[start:end],y_train[start:end]) 
	mini_batches=[(x_train[i*mini_batch_size:min(i*mini_batch_size+mini_batch_size,len(x_train))],
		y_train[i*mini_batch_size:min(i*mini_batch_size+mini_batch_size,len(y_train))])for i in range(batch_size)]

	epoch=3
	print(x.shape,mini_batches[0][0].shape)
	print(y_.shape,mini_batches[0][1].shape)
	print(x_test.shape)
	print(y_test.shape)

	with tf.Session() as sess:
		sess.run(init_op)
		for i in range(epoch):
			print("第0次迭代".format(i))
			for mini_batch in mini_batches:				
				sess.run(train_step,feed_dict={x:mini_batch[0],y_:mini_batch[1]})
		sess.run(accuracy,feed_dict={x:x_test,y_:y_test})
