from distutils.core import setup
from subprocess import getoutput
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from os.path import (dirname,
                     realpath,
                     join)

util_dir = dirname(realpath(__file__))
src_dir = join(dirname(util_dir),
               'src')
root_env = getoutput("conda info | grep \"root environment :\""
                     " | awk '{print $4}'")
python_header_dir = join(root_env,
                         'pkgs/python-3.4.3-0/include/python3.4m')

ext_modules = [Extension('feature_extraction',
                         [join(src_dir,
                               "feature_extraction.pyx")],
                         include_dirs=[python_header_dir]),
               Extension('datasets',
                         [join(util_dir,
                               "datasets.pyx")],
                         include_dirs=[python_header_dir]),
               Extension('mongodb',
                         [join(util_dir,
                               "mongodb.pyx")],
                         include_dirs=[python_header_dir])]

setup(
    name = "reviewer experience prediction system",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
)