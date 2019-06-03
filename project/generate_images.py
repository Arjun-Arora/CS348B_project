import copy
import numpy as np
import subprocess
import sys

NUM_IMAGES = 100
pbrt_dir = sys.argv[1]
pbrt_scene = sys.argv[2]
exr_filename_idx = int(sys.argv[3]) - 1
lookat_idx = int(sys.argv[4]) - 1
sample_idx = int(sys.argv[5]) - 1

def save_pbrt_and_run(i, new_pbrt_file, pixel_samples):
	extension = "_" + str(i) + "_" + str(pixel_samples)

	idx = new_pbrt_file[exr_filename_idx].find(".exr")
	new_pbrt_file[exr_filename_idx] = new_pbrt_file[exr_filename_idx][:idx] + extension + ".exr\"\n"

	sample = new_pbrt_file[sample_idx].split()
	sample[-1] = str(pixel_samples)
	new_pbrt_file[sample_idx] = " ".join(sample) + "\n"

	path = pbrt_dir + pbrt_scene + extension + ".pbrt"
	f = open(path, "w")
	for line in new_pbrt_file:
		f.write(line)
	f.close()

	subprocess.run(["pbrtBuild/pbrt", path])
	subprocess.run(["mv", pbrt_scene + extension + ".exr", pbrt_dir])
	if pixel_samples == 64:
		subprocess.run(["mv", "feature_map.txt", pbrt_dir + "feature_map" + extension + ".txt"])
	else:
		subprocess.run(["rm", "feature_map.txt"])

pbrt_file = []
with open(pbrt_dir + pbrt_scene + ".pbrt", "r") as f:
	for line in f:
		pbrt_file.append(line)

for i in range(NUM_IMAGES):
	new_pbrt_file = copy.deepcopy(pbrt_file)

	lookat = new_pbrt_file[lookat_idx].split()
	for j in range(1, 7):
		lookat[j] = "%.5f" % (float(lookat[j]) + np.random.uniform(-0.5, 0.5))
	new_pbrt_file[lookat_idx] = " ".join(lookat) + "\n"

	save_pbrt_and_run(i, copy.deepcopy(new_pbrt_file), 64)
	save_pbrt_and_run(i, copy.deepcopy(new_pbrt_file), 4096)
