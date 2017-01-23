import os
from shutil import copyfile
import pickle
import numpy as np
import matlab.engine

"""
To be used with toolkit accessible at:
- http://www.peterkovesi.com/studentprojects/libor/sourcecode.html

Python 2.7.12 & Matlab R2015b

Place the file within the folder containing 'createiristemplate.m'.

"""

# Do a walk and copy all jpeg images to a single directory.
file_paths = []
for root, dirs, files in os.walk("CASIA"):
    file_paths.extend(os.path.join(root, name) for name in files if 'jpg' in name)

for path in file_paths:
    copyfile(path, "all_images/"+path.split('/')[-1])

# Start matlab engine and configure pathing.
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('Segmentation'))
eng.addpath(eng.genpath('Normal_encoding'))
eng.addpath(eng.genpath('Matching'))
eng.addpath(eng.genpath('all_images'))

# Run segmentation and cache data for later use.
# for img in [f for f in os.listdir('all_images') if 'jpg' in f]:
#         eng.createiristemplate(img, nargout=2)

if __name__ == '__main__':
    iris_templates = {}
    for img in [f for f in os.listdir('all_images') if 'jpg' in f]:
        t, m = eng.createiristemplate(img, nargout=2)
        t = np.array(t._data, dtype=np.int).reshape(t.size[::-1]).T
        m = np.array(m._data, dtype=np.int).reshape(m.size[::-1]).T
        iris_templates[img] = t, m

    """ Save iris templates for later program runs. """
    with open('iris_templates.dat', "wb") as f:
        pickle.dump(iris_templates, f)
