import numpy as np

def loadMotorTaskDesign():
	design_matrix_path = "./data/motor/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2, 4, 6, 8, 10])
	return design_matrix

def loadEmotionTaskDesign():
	design_matrix_path = "./data/emotion/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2])
	return design_matrix

def loadGamblingTaskDesign():
	design_matrix_path = "./data/gambling/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2])
	return design_matrix

def loadLanguageTaskDesign():
	design_matrix_path = "./data/language/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2])
	return design_matrix

def loadRelationalTaskDesign():
	design_matrix_path = "./data/relational/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2])
	return design_matrix

def loadSocialTaskDesign():
	design_matrix_path = "./data/social/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2])
	return design_matrix

def loadWMTaskDesign():
	design_matrix_path = "./data/WM/design.mat"
	design_matrix = np.loadtxt(design_matrix_path,
							   delimiter="\t",
							   skiprows=5,
							   dtype=float,
							   usecols=[0, 2, 4, 6, 8, 10, 12, 14])
	return design_matrix