#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def generate_sinusoids_1d (lb, ub, num_pts, y_offset):
    # generate multi-dimensional sinusoid
    X = np.random.rand(num_pts,1) * (ub-lb) + lb
    scale = 2 * np.pi / (ub-lb)
    Y = np.cos(scale * X) + y_offset + 0.1 * np.random.randn(num_pts,1)

    return X, Y

def generate_sinusoids_2d (lb, ub, num_pts, y_offset):
    # generate multi-dimensional sinusoid
    X = np.random.rand(num_pts,1) * (ub-lb) + lb
    Y = np.zeros((num_pts,2))
    scale = 2 * np.pi / (ub-lb)
    Y[:,0] = np.squeeze(np.cos(scale * X) + y_offset[0] + 0.1 * np.random.randn(num_pts,1))
    Y[:,1] = np.squeeze(np.sin(scale * X) + y_offset[1] + 0.1 * np.random.randn(num_pts,1))

    return X, Y

def plot_sinusoid_dataset(X, Y, title_string, figure_name=None):
	if figure_name == None:
		fig = plt.figure(figsize=(10, 8))
	else:
		fig = plt.figure(figure_name, figsize=(10, 8))
		plt.clf()

	# one variable
	if Y.shape[1] == 1:
		plt.scatter(X, Y, 50)
		plt.title(title_string)
	
	elif Y.shape[1] == 2:
		assert(Y.shape[1] == 2)
		plt.subplot(211)
		plt.scatter(X, Y[:,0], 50)
		plt.title(title_string + ', 1st variable')

		plt.subplot(212)
		plt.scatter(X, Y[:,1], 50)
		plt.title(title_string + ', 2nd variable')
	
	plt.draw()
	plt.pause(0.0001)
	# plt.show(block='false')

def plot_sinusoid_dataset_compare(X, Y, X_test, Y_test, title_string='Testing', if_new_figure=1):
	if if_new_figure == 1:
		fig = plt.figure(figsize=(10, 8))
	else:
		plt.clf()

	# one variable
	if Y.shape[1] == 1:
		plt.scatter(X, Y, 50)
		plt.plot(X_test, Y_test, 'r-', linewidth=2)
		plt.title(title_string)
	
	elif Y.shape[1] == 2:
		assert(Y.shape[1] == 2)
		plt.subplot(211)
		plt.scatter(X, Y[:,0], 50)
		plt.plot(X_test, Y_test[:,0], 'r-', linewidth=2)
		plt.title(title_string + ', 1st variable')

		plt.subplot(212)
		plt.scatter(X, Y[:,1], 50)
		plt.plot(X_test, Y_test[:,1], 'r-', linewidth=2)
		plt.title(title_string + ', 2nd variable')
	



if __name__ == "__main__":
	print('hello_world from generate_sinusoids.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})
	# dataset_name = "/sinusoid1D"; func = generate_sinusoids_1d; 
	# y_offset = np.array([1.0]); lb = np.array([1]); ub = np.array([2]); num_pts = 100
	dataset_name = "/sinusoid2D"; func = generate_sinusoids_2d; 
	y_offset = np.array([1.0, 2.0]); lb = np.array([1]); ub = np.array([2]); num_pts = 100
	# load or create training data
	try:
		dataset = pickle.load(open(file_dir+dataset_name+"_dataset_train.p","rb"))
		X = dataset.X
		Y = dataset.Y
		assert(0)
	except:
		print('loading training data failed (should check)')
		X, Y = func(lb, ub, num_pts, y_offset)
		dataset = []
		dataset.append(X)
		dataset.append(Y)
		pickle.dump(dataset, open(file_dir+dataset_name+"_dataset_train.p", "wb"))

	# load or create visualization data 
	try: 
		X_vis = pickle.load(open(file_dir+dataset_name+"_dataset_vis.p", "rb"))
		assert(0)
	except:
		print('loading visualization data failed (should check)')
		X_vis = np.linspace(lb, ub, 50).reshape((50,1))
		pickle.dump(X_vis, open(file_dir+dataset_name+"_dataset_vis.p", "wb"))
	

	# plot visualization data
	Y_vis = np.ones((X_vis.shape[0],1), dtype='int')
	plot_sinusoid_dataset(X_vis, Y_vis, 'visulaization data')


	# plot training data
	plot_sinusoid_dataset(X, Y, 'training data')

	plt.show()




		


