import numpy as np
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

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
			print(i + 1)
			plot_stat_map(cur_img, display_mode="z",
						  cut_coords=cut_coords, colorbar=colorbar, annotate=annotate)
			show()