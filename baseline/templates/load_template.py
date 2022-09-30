import numpy as np
import os

from baseline.utils import thresholding

data_path = os.path.dirname(os.path.abspath(__file__))

def loadRSNTemplate10(threshold=False):
	template = np.load(data_path + "/data/ICA_template_10.npy")
	if threshold:
		template = thresholding(template)
	return template

def loadMotorTemplate(threshold=False):
	template = np.load(data_path + "/data/MOTOR_maps.npy")[:6, :]
	if threshold:
		template = thresholding(template)
	return template

def loadEmotionTemplate(threshold=False):
	template = np.load(data_path + "/data/EMOTION_maps.npy")[:2, :]
	if threshold:
		template = thresholding(template)
	return template

def loadGamblingTemplate(threshold=False):
	template = np.load(data_path + "/data/GAMBLING_maps.npy")[:2, :]
	if threshold:
		template = thresholding(template)
	return template

def loadLanguageTemplate(threshold=False):
	template = np.load(data_path + "/data/LANGUAGE_maps.npy")[:2, :]
	if threshold:
		template = thresholding(template)
	return template

def loadRelationalTemplate(threshold=False):
	template = np.load(data_path + "/data/RELATIONAL_maps.npy")[:2, :]
	if threshold:
		template = thresholding(template)
	return template

def loadSocialTemplate(threshold=False):
	template = np.load(data_path + "/data/SOCIAL_maps.npy")[:2, :]
	if threshold:
		template = thresholding(template)
	return template

def loadWMTemplate(threshold=False):
	template = np.load(data_path + "/data/WM_maps.npy")[:8, :]
	if threshold:
		template = thresholding(template)
	return template
