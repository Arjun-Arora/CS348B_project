import numpy as np
import os
import sys

DATA_DIR = sys.argv[1]

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        f = open(DATA_DIR + filename)
        lines = [line for line in f]
        dimy, dimx = tuple([int(x) for x in lines[0].split()])
        normal_x = np.zeros((dimx, dimy), dtype=np.float32)
        for i in range(dimy):
            normal_x[:, i] = [float(x) for x in lines[i + 1].split()]
        normal_y = np.zeros((dimx, dimy), dtype=np.float32)
        for i in range(dimy):
            normal_y[:, i] = [float(x) for x in lines[dimy + i + 1].split()]
        depth = np.zeros((dimx, dimy), dtype=np.float32)
        for i in range(dimy):
            depth[:, i] = [float(x) for x in lines[2*dimy + i + 1].split()]
        albedo = np.zeros((dimx, dimy, 3), dtype=np.float32)
        for i in range(dimy):
            albedo[:, i, :] = np.array([[float(x) for x in triple.split(',')] for triple in lines[3*dimy + i + 1].split()])
        x = np.dstack((np.expand_dims(normal_x, axis=2), \
                       np.expand_dims(normal_y, axis=2), \
                       np.expand_dims(depth, axis=2), \
                       albedo))
        np.save(DATA_DIR + filename[:-4] + ".npy", x)
