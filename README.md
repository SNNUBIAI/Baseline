# Baseline
Baseline is a toolbox to use general linear model (GLM), sparse dictionary learning (SDL) and independent component analysis (ICA) for mapping functional brain networks (FBNs, sometimes called functional domain or spatial pattern in some papers).
The GLM, SDL and ICA are widely accepted and used in clinical practice, and are very suitable as the baseline methods for comparision with other methods.

We provide a benchmark to evaluate the quality of the functional brain networks generated by different methods which incorporates 10 widely accepted resting-state networks (RSN) templates and 7 task-state templates 
(emotion, gambling, language, motor, relational, social and working memory) generated by GLM.

- RSN templates is based on `S.M. Smith, P.T. Fox, K.L. Miller, D.C. Glahn, P.M. Fox, C.E. Mackay, N. Filippini, K.E. Watkins, R. Toro, A.R. Laird, and C.F. Beckmann. Correspondence of the brain's functional architecture during activation and rest. Proc Natl Acad Sci USA (PNAS), 106(31):13040-13045, 2009.` [download link](https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/)
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