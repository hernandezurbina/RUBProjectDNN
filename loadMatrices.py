import sys
import numpy as np

fileName = sys.argv[1]
with open(fileName, "r") as f:
	mat = np.loadtxt(f, dtype="float32", delimiter=",")

print(np.shape(mat))




