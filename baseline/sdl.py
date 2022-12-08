from nilearn.decomposition import DictLearning
import numpy as np

from baseline.masker import Masker

class SDL(Masker):
	def __init__(self, mask_path, data_path,
				 n_components=20, alpha=10, random_state=0, n_jobs=-1):
		super(SDL, self).__init__(mask_path=mask_path)
		self.mask_path = mask_path
		if isinstance(data_path, str) and data_path.endswith(".npy"):
			data_path = self.img2NiftImage(data_path)
		self.data_path = data_path
		self.n_components = n_components
		self.random_state = random_state
		self.alpha = alpha
		self.dict_learning = DictLearning(mask=self.mask_path,
										n_components=self.n_components,
										memory="nilearn_cache", memory_level=2,
										alpha=self.alpha,
										random_state=self.random_state,
										n_jobs=n_jobs)

	def fit(self, data_path=None):
		if data_path is not None:
			if isinstance(data_path, str) and data_path.endswith(".npy"):
				data_path = self.img2NiftImage(data_path)
			elif isinstance(data_path, np.ndarray):
				data_path = self.img2NiftImage(data_path)
			else:
				raise ValueError("data_path must be a numpy ndarray vector or the npy path or the nifti nii")
			self.dict_learning.fit(data_path)
		else:
			self.dict_learning.fit(self.data_path)
		return self

	def get_3d(self):
		return self.dict_learning.components_img_

	def get_2d(self):
		return self.transform(self.get_3d().get_fdata())

	def setNcomponents(self, n_components):
		self.n_components = n_components
		self.dict_learning = DictLearning(mask=self.mask_path,
						n_components=self.n_components,
						memory="nilearn_cache", memory_level=2,
						alpha=self.alpha,
						random_state=self.random_state, n_jobs=-1)

	def fit_transform(self, data_path):
		self.fit(data_path)
		return self.get_2d()