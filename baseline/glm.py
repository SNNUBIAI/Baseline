import numpy as np
import statsmodels.api as sm

from copy import deepcopy

class GLM:
	def __init__(self, fmri_signals, design_matrix, p=0.01):
		self.p = p
		self.fmri_signals = fmri_signals
		self.design_matrix = design_matrix
		self.coef_ = None
		self.p_ = None

	def fit(self):
		design = sm.add_constant(self.design_matrix)
		pvalue_list = []
		coef_list = []
		for i in range(self.fmri_signals.shape[1]):
			est = sm.OLS(self.fmri_signals[:, i], design)
			est = est.fit()
			pvalue = est.pvalues.reshape(1, -1)
			coef = est.params.reshape(1, -1)
			pvalue_list.append(pvalue)
			coef_list.append(coef)
		pvalue_list = np.concatenate(pvalue_list, axis=0)
		coef_list = np.concatenate(coef_list, axis=0)
		self.p_ = pvalue_list[:, 1:].T
		self.coef_ = coef_list[:, 1:].T

		return self

	def setPvalue(self, p):
		self.p = p

	def get_coef(self):
		coef = deepcopy(self.coef_)
		coef[self.p_ > self.p] = 0
		return coef

	def __repr__(self):
		return "GLM(p={})".format(self.p)