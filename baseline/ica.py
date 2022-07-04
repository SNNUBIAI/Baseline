from nilearn.input_data import NiftiMasker
from nilearn.decomposition import CanICA


class ICA:
    def __init__(self, mask_path, data_path,
                 n_components, random_state):
        self.mask_path = mask_path
        self.data_path = data_path
        self.n_components = n_components
        self.random_state = random_state
        self.Ica_3d = None

    def fit(self):
        canica = CanICA(mask=self.mask_path,
                        n_components=self.n_components,
                        memory="nilearn_cache", memory_level=2,
                        random_state=self.random_state)
        canica.fit(self.data_path)

        self.Ica_3d = canica.components_img_

        return self

    def get_2d(self):
        masker = NiftiMasker(mask_img=self.mask_path,
                             standardize=False)
        masker.fit()
        Ica_2d = masker.transform(self.Ica_3d)

        return Ica_2d
