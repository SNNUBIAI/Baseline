# Baseline
GLM, SDL, ICA for mapping brain function

## Requirement
- statsmodels
- numpy

## Example
```python
from baseline.glm import GLM

design_matrix_path = "/home/public/ExperimentData/HCP900/HCP_data/design/emotion/design.mat"
design_matrix = np.loadtxt(design_matrix_path, delimiter="\t", skiprows=5, dtype=float, usecols=[0, 1, 2, 3])
fmri_signals = np.load("/home/public/ExperimentData/HCP900/HCP_data/SINGLE/EMOTION_sub_0.npy")

glm = GLM(fmri_signals=fmri_signals, design_matrix=design_matrix, p=0.001)
glm.fit()

coef = glm.get_coef()
```
