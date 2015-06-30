from sys import exit
from os import listdir
from shutil import copy
from getpass import getuser
from os.path import (dirname,
                     realpath,
                     join,
                     exists)
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
# Try to guess the location of the conda installation
if not root_env:
    root_env = '/home/{}/conda'.format(getuser())
python_header_dir = join(root_env,
                         'pkgs/python-3.4.3-0/include/python3.4m')

ext_names = {'feat': 'feature_extraction',
             'data': 'datasets',
             'db': 'mongodb'}

ext_modules = [Extension('feature_extraction',
                         [join(src_dir,
                               "{0}.pyx".format(ext_names['feat']))],
                         include_dirs=[python_header_dir]),
               Extension('datasets',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['data']))],
                         include_dirs=[python_header_dir]),
               Extension('mongodb',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['db']))],
                         include_dirs=[python_header_dir])]

setup(
    name = "reviewer experience prediction system",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
)

# Copy files from build directory
if not exists(build_dir):
    exit('Build directory does not exist. Exiting.')
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