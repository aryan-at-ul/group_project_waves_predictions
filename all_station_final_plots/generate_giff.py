import imageio
import os
import sys

data_dir = "./M1/period"
all_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(('.png')):
            print(root,dirs,filename)
            all_files.append(root + '/' + filename)





images = []
for filename in all_files:
    images.append(imageio.imread(filename))
imageio.mimsave('m1_period_movie.gif', images,fps = 1)
