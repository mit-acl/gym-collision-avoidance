#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_sinusoids_sum_1out (lb, ub, num_pts, y_offset, if_add_noise=True):
    # generate multi-dimensional sinusoid
    x1 = np.linspace(lb, ub, num_pts+1)
    x2 = np.linspace(lb, ub, num_pts+1)
    num_grid_pts = (num_pts+1) ** 2
    X1, X2 = np.meshgrid(x1, x2)
    X1 = np.reshape(X1, (num_grid_pts,1))
    X2 = np.reshape(X2, (num_grid_pts,1))
    X = np.hstack((X1, X2))

    scale = 2 * np.pi / (ub-lb)
    if if_add_noise == True:
    	# X += 0.1 * np.random.randn(num_grid_pts,2)
    	Y = np.sin(scale * X1) + np.sin(scale * X2) + y_offset + 0.1 * np.random.randn(num_grid_pts,1)
    else:
    	Y = np.sin(scale * X1) + np.sin(scale * X2) + y_offset


    # print X.shape
    # print Y.shape
    # raw_input()

    return X, Y

# def generate_sinusoids_sum_2out (lb, ub, num_pts, y_offset):
#     # generate multi-dimensional sinusoid
#     X = np.random.rand(num_pts,1) * (ub-lb) + lb
#     Y = np.zeros((num_pts,2))
#     scale = 2 * np.pi / (ub-lb)
#     Y[:,0] = np.squeeze(np.cos(scale * X) + y_offset[0] + 0.1 * np.random.randn(num_pts,1))
#     Y[:,1] = np.squeeze(np.sin(scale * X) + y_offset[1] + 0.1 * np.random.randn(num_pts,1))

#     return X, Y

def plot_sinusoid_dataset(X, Y, title_string, figure_name=None):
	if figure_name == None:
		fig = plt.figure(figsize=(10, 8))
	else:
		fig = plt.figure(figure_name, figsize=(10, 8))
		plt.clf()

	# one variable
	if Y.shape[1] == 1:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0], X[:,1], Y) #, ms=50)
		plt.title(title_string)
	
	elif Y.shape[1] == 2:
		assert(Y.shape[1] == 2)
		ax1 = fig.add_subplot(211, projection='3d')
		ax1.scatter(X[:,0], X[:,1], Y[:,0]) #, ms=50)
		plt.title(title_string + ', 1st variable')

		ax2 = fig.add_subplot(212, projection='3d')
		ax2.scatter(X[:,0], X[:,1], Y[:,1]) #, ms=50)
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
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0], X[:,1], Y)
		X1_test_mat, X2_test_mat, Y_test_mat = array2mat(X_test, Y_test)
		ax.plot_wireframe(X1_test_mat, X2_test_mat, Y_test_mat, color='r', linewidth=2)
		plt.title(title_string)
	
	elif Y.shape[1] == 2:
		assert(Y.shape[1] == 2)
		ax1 = fig.add_subplot(211, projection='3d')
		ax1.scatter(X[:,0], X[:,1], Y[:,0])
		X1_test_mat, X2_test_mat, Y_test_mat = array2mat(X_test, Y_test[:,0])
		ax1.plot_wireframe(X1_test_mat, X2_test_mat, Y_test_mat, color='r', linewidth=2)
		plt.title(title_string + ', 1st variable')

		ax2 = fig.add_subplot(212, projection='3d')
		ax2.scatter(X[:,0], X[:,1], Y[:,1])
		X1_test_mat, X2_test_mat, Y_test_mat = array2mat(X_test, Y_test[:,1])
		ax2.plot_wireframe(X1_test_mat, X2_test_mat, Y_test_mat, color='r', linewidth=2)
		plt.title(title_string + ', 2nd variable')

def array2mat(X,Y):
	num_grid_pts = X.shape[0]
	EPS = 1e-5
	for i in xrange(1,num_grid_pts):
		if abs(X[i,0] - X[0,0]) < EPS:
			x_grid_num = i
			break
	# print x_grid_num
	X1 = np.reshape(X[:,0], (x_grid_num, -1))
	X2 = np.reshape(X[:,1], (x_grid_num, -1))
	Y = np.reshape(Y, (x_grid_num, -1))
	return X1, X2, Y



if __name__ == "__main__":
	print('hello_world from generate_sinusoids.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})
	dataset_name = "/sinusoid_sum_1out"; func = generate_sinusoids_sum_1out ; 
	y_offset = np.array([1.0]); lb = np.array([1]); ub = np.array([2]); num_pts = 20
	# dataset_name = "/sinusoid_sum_2out"; func = generate_sinusoids_sum_2out ; 
	# y_offset = np.array([1.0, 2.0]); lb = np.array([1]); ub = np.array([2]); num_pts = 20
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
		X_vis, Y_vis = func(lb, ub, num_pts, y_offset, if_add_noise=False)
		pickle.dump(X_vis, open(file_dir+dataset_name+"_dataset_vis.p", "wb"))
	

	# plot visualization data
	# Y_vis = np.ones((X_vis.shape[0],1), dtype='int')
	# plot_sinusoid_dataset(X_vis, Y_vis, 'visulaization data')


	# plot training data
	plot_sinusoid_dataset(X, Y, 'training data')
	plot_sinusoid_dataset_compare(X, Y, X_vis, Y_vis, title_string='Testing', if_new_figure=1)

	plt.show()