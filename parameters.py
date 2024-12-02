import numpy as np
import cv2

camera_matrix = np.load('intrinsic_extrinsic/camera_matrix.npy')
dist_coeffs = np.load('intrinsic_extrinsic/dist_coefficients.npy')

# Configurar o detector de ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
