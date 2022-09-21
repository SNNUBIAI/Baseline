# Baseline
GLM, SDL, ICA for mapping brain function

## Requirement
- nilearn
- statsmodels
- numpy

## Install
```shell script
git clone https://github.com/SNNUBIAI/Baseline.git
cd Baseline
pip install .
```

## Example
- GLM
```python
from baseline.glm import GLM
from baseline.templates import loadEmotionTaskDesign

design_matrix = loadEmotionTaskDesign()
fmri_signals = np.load("/home/public/ExperimentData/HCP900/HCP_data/SINGLE/EMOTION_sub_0.npy")

glm = GLM(fmri_signals=fmri_signals, design_matrix=design_matrix, p=0.001)
glm.fit()

coef = glm.get_coef()
```

- Load Template and task design curve
```python
from baseline.templates import loadMotorTemplate
from baseline.templates import loadMotorTaskDesign

template = loadMotorTemplate()
task_design = loadMotorTaskDesign()
```