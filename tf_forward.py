import matplotlib
matplotlib.use('TkAgg')
import sys
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import math
import random
import json


def getActivations(layer,stimuli):
	units = sess.run(layer, feed_dict=feed_dict)
	print(np.shape(units))
	plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    # plt.figure(1, figsize=(20,20))
    fig = plt.figure()
    n_columns = 10
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.axis('off')
        # plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.imshow(units[0,:,:,i], interpolation="nearest")
    fig.savefig('temp.png')

def getFilterMatrices():
	# FIRST CONV LAYER
	units = sess.run(graph.get_tensor_by_name("import/conv1_2/Conv2D:0"), feed_dict=feed_dict)
	numFilters = units.shape[3]
	rndFilter = random.randrange(numFilters)
	print("First Conv Layer")
	print("Chosen filter: ", rndFilter)
	# print(np.shape(units))
	print()
	with open("CNN_layer1.csv","wb") as f:
		np.savetxt(f, units[0, :, :, rndFilter],fmt="%2.2f", delimiter=",")
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(units[0,:,:,rndFilter], interpolation="nearest")
	fig.savefig('CNN_layer1.png')

	# SECOND CONV LAYER
	units = sess.run(graph.get_tensor_by_name("import/conv2_2/Conv2D:0"), feed_dict=feed_dict)
	numFilters = units.shape[3]
	rndFilter = random.randrange(numFilters)
	print("Second Conv Layer")
	print("Chosen filter: ", rndFilter)
	# print(np.shape(units))
	print()
	with open("CNN_layer2.csv","wb") as f:
		np.savetxt(f, units[0, :, :, rndFilter],fmt="%2.2f", delimiter=",")
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(units[0,:,:,rndFilter], interpolation="nearest")
	fig.savefig('CNN_layer2.png')


	# THIRD CONV LAYER
	units = sess.run(graph.get_tensor_by_name("import/conv3_3/Conv2D:0"), feed_dict=feed_dict)
	numFilters = units.shape[3]
	rndFilter = random.randrange(numFilters)
	print("Third Conv Layer")
	print("Chosen filter: ", rndFilter)
	# print(np.shape(units))
	print()
	with open("CNN_layer3.csv","wb") as f:
		np.savetxt(f, units[0, :, :, rndFilter],fmt="%2.2f", delimiter=",")
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(units[0,:,:,rndFilter], interpolation="nearest")
	fig.savefig('CNN_layer3.png')		

	# FOURTH CONV LAYER
	units = sess.run(graph.get_tensor_by_name("import/conv4_3/Conv2D:0"), feed_dict=feed_dict)
	numFilters = units.shape[3]
	rndFilter = random.randrange(numFilters)
	print("Fourth Conv Layer")
	print("Chosen filter: ", rndFilter)
	# print(np.shape(units))
	print()
	with open("CNN_layer4.csv","wb") as f:
		np.savetxt(f, units[0, :, :, rndFilter],fmt="%2.2f", delimiter=",")
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(units[0,:,:,rndFilter], interpolation="nearest")
	fig.savefig('CNN_layer4.png')


	# FIFTH CONV LAYER
	units = sess.run(graph.get_tensor_by_name("import/conv5_3/Conv2D:0"), feed_dict=feed_dict)
	numFilters = units.shape[3]
	rndFilter = random.randrange(numFilters)
	print("Fifth Conv Layer")
	print("Chosen filter: ", rndFilter)
	# print(np.shape(units))
	print()
	with open("CNN_layer5.csv","wb") as f:
		np.savetxt(f, units[0, :, :, rndFilter],fmt="%2.2f", delimiter=",")
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(units[0,:,:,rndFilter], interpolation="nearest")
	fig.savefig('CNN_layer5.png')


random.seed()

with open("vgg16.tfmodel", mode='rb') as f:
	fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print()
print("Graph loaded from disk")

graph = tf.get_default_graph()

img = utils.load_image(sys.argv[1])

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print()
	print("Variables initialized")

	batch = img.reshape((1, 224, 224, 3))
	assert batch.shape == (1, 224, 224, 3)

	feed_dict = { images: batch }

	prob_tensor = graph.get_tensor_by_name("import/prob:0")
	prob = sess.run(prob_tensor, feed_dict=feed_dict)

	print()
	print("-"*100)
	# To plot the feature map of a particular layer uncomment the line below
	# getActivations(graph.get_tensor_by_name("import/conv1_1/Relu:0"),img)
	print("Generating output files with matrices from convolution layers")
	print()
	getFilterMatrices()
	print("-"*100)

utils.print_prob(prob[0])


