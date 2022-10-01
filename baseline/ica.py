from sklearn.decomposition import fastica
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state
from joblib import delayed, Parallel
from scipy.stats import scoreatpercentile
from tqdm import trange
import numpy as np

from operator import itemgetter

from baseline.masker import Masker

class canICA:
	def __init__(self,
				 n_components=20,
				 mask_img=None,
				 n_jobs=-1,
				 random_state=666):
		if mask_img is not None:
			self.masker = Masker(mask_path=mask_img)
		else:
			self.masker = None
		self.mask_img = mask_img
		self.n_components = n_components
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.components_ = None
		self.components_img_ = None

	def fit(self, fmri_data):
		if type(fmri_data) == str:
			if fmri_data.endswith(".npy"):
				fmri_data = np.load(fmri_data)
			else:
				raise ValueError("fmri_data must be numpy.ndarray."
								 "You provided {}".format(fmri_data))

		S = np.sqrt(np.sum(fmri_data ** 2, axis=1))
		S[S == 0] = 1
		fmri_data /= S[:, np.newaxis]
		components_, variance_, _ = randomized_svd(fmri_data.T,
												   n_components=self.n_components,
												   transpose=True,
												   random_state=self.random_state,
												   n_iter=3)
		fmri_data *= S[:, np.newaxis]
		self.components_ = components_.T
		seeds = check_random_state(self.random_state).randint(np.iinfo(np.int32).max, size=10)

		results = Parallel(n_jobs=self.n_jobs, verbose=0)(
			delayed(fastica)(components_.astype(np.float64), whiten=True, fun='cube', random_state=seed)
			for seed in seeds)
		ica_maps_gen_ = (result[2].T for result in results)
		ica_maps_and_sparsities = ((ica_map,
									np.sum(np.abs(ica_map), axis=1).max())
								   for ica_map in ica_maps_gen_)

		ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

		# auto thresholding
		abs_ica_maps = np.abs(ica_maps)
		percentile = 100. - (100. / len(ica_maps))
		threshold = scoreatpercentile(abs_ica_maps, percentile)
		ica_maps[abs_ica_maps < threshold] = 0.

		self.components_ = ica_maps.astype(self.components_.dtype)
		for component in self.components_:
			if component.max() < -component.min():
				component *= -1

		if self.masker is not None:
			self.components_img_ = self.masker.img2NiftImage(self.components_)
		return self

	def to_nii(self, path):
		if self.masker is not None:
			data = self.masker.img2NiftImage(self.components_)
			data.to_filename(path)
		else:
			raise ValueError("Do not have any masker to transform back to 3D volume.")

class SlideWindowICA:
	def __init__(self,
				 fmri_data,
				 window_size=40,
				 stride=1,
				 n_components=20,
				 mask_img=None,
				 n_jobs=-1,
				 random_state=666):
		self.ica = canICA(n_components=n_components,
						  mask_img=mask_img,
						  n_jobs=n_jobs,
						  random_state=random_state)

		self.window_size = window_size
		self.stride = stride

		if type(fmri_data) == str:
			if fmri_data.endswith(".npy"):
				fmri_data = np.load(fmri_data)
			else:
				raise ValueError("fmri_data must be numpy.ndarray."
								 "You provided {}".format(fmri_data))
		self.fmri_data = fmri_data

		self.time_step = fmri_data.shape[0]
		self.sliding_times = (self.time_step - self.window_size) // self.stride + 1
		self.components_list_ = []

	def fit(self):
		for i in trange(0, self.sliding_times, self.stride):
			self.ica.fit(self.fmri_data[i:i+self.window_size, :])
			self.components_list_.append(self.ica.components_)
		return self.get_components()

	def setWindowSize(self, window_size):
		self.window_size = window_size
		self.sliding_times = (self.time_step - self.window_size) // self.stride + 1

	def setStride(self, stride):
		self.stride = stride
		self.sliding_times = (self.time_step - self.window_size) // self.stride + 1

	def setfMRIData(self, fmri_data):
		if type(fmri_data) == str:
			if fmri_data.endswith(".npy"):
				fmri_data = np.load(fmri_data)
			else:
				raise ValueError("fmri_data must be numpy.ndarray."
								 "You provided {}".format(fmri_data))
		self.fmri_data = fmri_data
		self.time_step = fmri_data.shape[0]
		self.sliding_times = (self.time_step - self.window_size) // self.stride + 1
		self.components_list_ = []

	def get_components(self):
		assert len(self.components_list_) != 0
		return np.concatenate(self.components_list_, axis=0)