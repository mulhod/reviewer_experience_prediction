from os import listdir
from sys import exit
from shutil import copy
from os.path import (dirname,
                     realpath,
                     join)
from subprocess import getoutput
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

util_dir = dirname(realpath(__file__))
src_dir = join(dirname(util_dir),
               'src')
build_dir = join(dirname(util_dir),
                 'build')

# Hackish way of doing this. Find better way...
root_env = getoutput("conda info | grep \"root environment :\""
                     " | awk '{print $4}'")
python_header_dir = join(root_env,
                         'pkgs/python-3.4.3-0/include/python3.4m')

ext_names = {'feat_ext': 'feature_extraction',
             'data': 'datasets',
             'mongo': 'mongodb'}

ext_modules = [Extension('feature_extraction',
                         [join(src_dir,
                               "{0}.pyx".format(ext_names['feat_ext']))],
                         include_dirs=[python_header_dir]),
               Extension('datasets',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['data']))],
                         include_dirs=[python_header_dir]),
               Extension('mongodb',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['mongo']))],
                         include_dirs=[python_header_dir])]

setup(
    name = "reviewer experience prediction system",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
)

# Copy files from build directory
build_libs_dir = [join(build_dir,
                       _dir) for _dir in listdir(build_dir) if
                  _dir.startswith('lib')]
if not build_libs_dir or len(build_libs_dir) > 1:
    exit('Could not find lib* directory in build/. Exiting.\n')
build_libs_dir = build_libs_dir[0]
exts = listdir(build_libs_dir)
for ext in exts:
    if ext.startswith('feature_extraction'):
        copy(join(build_libs_dir,
                  ext),
             join(src_dir,
                  '{}.so'.format(ext.split('.',
                                           1)[0])))
    else:
        copy(join(build_libs_dir,
                  ext),
             join(util_dir,
                  '{}.so'.format(ext.split('.',
                                           1)[0])))