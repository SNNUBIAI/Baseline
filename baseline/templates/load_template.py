import numpy as np
import os

data_path = os.path.dirname(os.path.abspath(__file__))

def loadRSNTemplate10():
	template = np.load(data_path + "/data/ICA_template_10.npy")
	return template

def loadMotorTemplate():
	# design_matrix_path = "./data/motor/design.mat"
	# design_matrix = np.loadtxt(design_matrix_path,
	# 						   delimiter="\t",
	# 						   skiprows=5,
	# 						   dtype=float,
	# 						   usecols=[0, 2, 4, 6, 8, 10])
	template = np.load(data_path + "/data/MOTOR_maps.npy")[:6, :]
	return template

def loadEmotionTemplate():
	template = np.load(data_path + "/data/EMOTION_maps.npy")[:2, :]
	return template

def loadGamblingTemplate():
	template = np.load(data_path + "/data/GAMBLING_maps.npy")[:2, :]
	return template

def loadLanguageTemplate():
	template = np.load(data_path + "/data/LANGUAGE_maps.npy")[:2, :]
	return template

def loadRelationalTemplate():
	template = np.load(data_path + "/data/RELATIONAL_maps.npy")[:2, :]
	return template

def loadSocialTemplate():
	template = np.load(data_path + "/data/SOCIAL_maps.npy")[:2, :]
	return template

def loadWMTemplate():
	template = np.load(data_path + "/data/WM_maps.npy")[:8, :]
	return template
