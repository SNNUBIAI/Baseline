from nilearn.decomposition import DictLearning

from baseline.masker import Masker

class SDL(Masker):
	def __init__(self, mask_path, data_path,
				 n_components, alpha, random_state):
		super(SDL, self).__init__(mask_path=mask_path)
		self.mask_path = mask_path
		self.data_path = data_path
		self.n_components = n_components
		self.random_state = random_state
		self.alpha = alpha
		self.dict_learning = DictLearning(mask=self.mask_path,
						n_components=self.n_components,
						memory="nilearn_cache", memory_level=2,
						alpha=self.alpha,
						random_state=self.random_state)

	def fit(self, data_path=None):
		if data_path is not None:
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
						random_state=self.random_state)

	def fit_transform(self, data_path):
		self.fit(data_path)
		return self.get_2d()