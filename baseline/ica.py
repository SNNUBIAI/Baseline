from nilearn.decomposition import CanICA

from baseline.masker import Masker

class ICA(Masker):
	def __init__(self, mask_path, data_path,
				 n_components, random_state):
		super(ICA, self).__init__(mask_path=mask_path)
		self.mask_path = mask_path
		self.data_path = data_path
		self.n_components = n_components
		self.random_state = random_state
		self.ica = CanICA(mask=self.mask_path,
						n_components=self.n_components,
						memory="nilearn_cache", memory_level=2,
						random_state=self.random_state)

	def fit(self, data_path=None):
		if data_path is not None:
			self.ica.fit(data_path)
		else:
			self.ica.fit(self.data_path)
		return self

	def get_3d(self):
		return self.ica.components_img_

	def get_2d(self):
		return self.transform(self.get_3d().get_fdata())

	def setNcomponents(self, n_components):
		self.n_components = n_components
		self.ica = CanICA(mask=self.mask_path,
						  n_components=self.n_components,
						  memory="nilearn_cache", memory_level=2,
						  random_state=self.random_state)

	def fit_transform(self, data_path):
		self.fit(data_path)
		return self.get_2d()