#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
# adopted from 6.883 MIT

def generate_spirals (n, k, curvature, jitter, x_center, y_center):
    # generates coordinates for k spirals, n points in each spiral
    #   X = n*k by 2 matrix of coordinates
    #   Y = label 1..k for each data point
    # additional arguments:
    #   curvature = curvature of the spirals (0 = no curvature)
    #   jitter = amount of noise (0 = no noise)
    #   [x_center, y_center] = center of the spirals
    X = np.zeros((n*k,2)) 
    Y = np.zeros((n*k,1))
    for j in xrange(0,k):
    	ind = np.arange(n*j,n*(j+1))
    	r = np.linspace(0.1, 1, n)
    	t = np.linspace(j*(2*np.pi/k), (j+curvature)*(2*np.pi/k),n) + np.random.randn(1,n)*jitter
    	t = np.squeeze(t.transpose())
    	X[ind,0] = x_center + r*np.sin(t)
    	X[ind,1] = y_center + r*np.cos(t)
    	Y[ind] = j

    return X, Y

def plot_spiral_datasetWrapper(X, scores, title_string, if_new_figure=1):
	Y = np.argmax(scores, axis = 1)
	plot_spiral_dataset(X, Y, title_string, if_new_figure)

def plot_spiral_dataset(X, Y, title_string, if_new_figure=1):
	k = np.amax(Y) + 1
	# plot data
	colors = np.floor(64/k)*Y
	if np.amax(colors) != 0:
		colors = colors / np.amax(colors)
	if if_new_figure == 1:
		fig = plt.figure(figsize=(10, 8))
	else:
		plt.cla()
	plt.scatter(X[:,0], X[:,1], 50, colors, cmap="rainbow")
	plt.title(title_string)
	plt.draw()
	plt.pause(0.0001)
	# plt.show(block='false')
	



if __name__ == "__main__":
	print('hello_world from generate_spirals.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})
	# load or create training data
	try:
		dataset = pickle.load(open(file_dir+"/spiral_dataset_train.p","rb"))
		X = dataset.X
		Y = dataset.Y
		k = np.amax(Y) + 1
		assert(0)
	except:
		print('loading training data failed (should check)')
		d = 2; k = 5; n = 50; curvature = 2; jitter = 0.3;
		x_center = 0.5
		y_center = 0.5
		X, Y = generate_spirals (n, k, curvature, jitter, x_center, y_center)
		dataset = []
		dataset.append(X)
		dataset.append(Y)
		pickle.dump(dataset, open(file_dir+"/spiral_dataset_train.p", "wb"))

	# load or create visualization data 
	try: 
		X_vis = pickle.load(open(file_dir+"/spiral_dataset_vis.p", "rb"))
	except:
		print('loading visualization data failed (should check)')
		X_vis_x = np.linspace(-1, 2, 50)
		X_vis_y = np.linspace(-1, 2, 50)
		X_vis_X, X_vis_Y = np.meshgrid(X_vis_x, X_vis_y)
		X_vis = np.vstack((X_vis_X.flatten(),\
						 X_vis_Y.flatten()) ).transpose()
		pickle.dump(X_vis, open(file_dir+"/spiral_dataset_vis.p", "wb"))
	

	# plot visualization data
	Y_vis = np.zeros((X_vis.shape[0],), dtype='int')
	plot_spiral_dataset(X_vis, Y_vis, 'visulaization data')


	# plot training data
	plot_spiral_dataset(X, Y, 'training data')

	plt.show()




		


