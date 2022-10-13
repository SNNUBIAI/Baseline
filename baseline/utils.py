import numpy as np
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

def IoU(n1, n2):
	"""
	:param n1: 1*N
	:param n2: 1*N
	:return: IoU
	"""
	intersect = np.logical_and(n1, n2)
	union = np.logical_or(n1, n2)
	I = np.count_nonzero(intersect)
	U = np.count_nonzero(union)
	return I / U

def flip(row):
	if np.sum(row > 0) < np.sum(row < 0):
		row *= -1
	return row

def thresholding(array):
	array1 = array

	for idx, row in enumerate(array):
		row = flip(row)
		row[row < 0] = 0
		T = np.amax(row) * 0.3
		row[np.abs(row) < T] = 0

		row = row / np.std(row)
		array1[idx, :] = row
	return array1

def plot_net(components_img,
			 cut_coords=(5, 10, 15, 20, 25, 30, 35, 40),
			 colorbar=True,
			 annotate=True):
	if annotate:
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z", title="index={}".format(i),
						  cut_coords=cut_coords, colorbar=colorbar, annotate=annotate)
			show()
	else:
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z",
						  cut_coords=cut_coords, colorbar=colorbar, annotate=annotate)
			show()

def evaluate_iou(img2d, template, verbose=1):
	iou = np.zeros((img2d.shape[0], template.shape[0]))
	for i in range(img2d.shape[0]):
		for j in range(template.shape[0]):
			iou[i, j] = IoU(template[j:j + 1, :], img2d[i:i + 1, :])
	if verbose:
		max_iou = iou.max(axis=0)
		max_index = iou.argmax(axis=0)
		print("Template\t", end="")
		for i in range(template.shape[0]):
			print("{}\t".format(i+1), end="")
		print("\nIoU\t\t", end="")
		for i in range(template.shape[0]):
			print("{:.4f}\t".format(max_iou[i]), end="")
		print("\nIndex\t\t", end="")
		for i in range(template.shape[0]):
			print("{}\t".format(max_index[i]), end="")
	return iou