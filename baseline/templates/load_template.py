import numpy as np

def loadRSNTemplate10():
	template = np.load("./data/ICA_template_10.npy")
	return template

def loadMotorTemplate():
	# design_matrix_path = "./data/motor/design.mat"
	# design_matrix = np.loadtxt(design_matrix_path,
	# 						   delimiter="\t",
	# 						   skiprows=5,
	# 						   dtype=float,
	# 						   usecols=[0, 2, 4, 6, 8, 10])
	template = np.load("./data/MOTOR_maps.npy")[:6, :]
	return template

def loadEmotionTemplate():
	template = np.load("./data/EMOTION_maps.npy")[:2, :]
	return template

def loadGamblingTemplate():
	template = np.load("./data/GAMBLING_maps.npy")[:2, :]
	return template

def loadLanguageTemplate():
	template = np.load("./data/LANGUAGE_maps.npy")[:2, :]
	return template

def loadRelationalTemplate():
	template = np.load("./data/RELATIONAL_maps.npy")[:2, :]
	return template

def loadSocialTemplate():
	template = np.load("./data/SOCIAL_maps.npy")[:2, :]
	return template

def loadWMTemplate():
	template = np.load("./data/WM_maps.npy")[:8, :]
	return template
