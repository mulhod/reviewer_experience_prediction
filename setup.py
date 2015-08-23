from sys import (exit,
                 stderr)
from os import (getcwd,
                listdir)
from shutil import copy
from os.path import (join,
                     exists,
                     dirname,
                     realpath)
from getpass import getuser
from subprocess import getoutput
from distutils.core import setup
from Cython.Build import cythonize
from setuptools import find_packages
from Cython.Distutils import build_ext
from distutils.extension import Extension

main_dir = dirname(realpath(__file__))
src_dir = join(main_dir,
               'src')
util_dir = join(main_dir,
                'util')
build_dir = join(main_dir,
                 'build')

def readme():
    with open('README.md') as f:
        return f.read()

# Hackish way of doing this. Find better way...
root_env = getoutput("conda info | grep \"root environment :\""
                     " | awk '{print $4}'")
# Try to guess the location of the conda installation
if not root_env:
    root_env = '/home/{}/conda'.format(getuser())
python_header_dir = join(root_env,
                         'pkgs/python-3.4.3-0/include/python3.4m')

ext_names = {'features': 'features',
             'data': 'datasets',
             'db': 'mongodb'}

ext_modules = [Extension('features',
                         [join(src_dir,
                               "{0}.pyx".format(ext_names['features']))],
                         include_dirs=[python_header_dir]),
               Extension('datasets',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['data']))],
                         include_dirs=[python_header_dir]),
               Extension('mongodb',
                         [join(util_dir,
                               "{0}.pyx".format(ext_names['db']))],
                         include_dirs=[python_header_dir])]

setup(name = 'Reviewer Experience Prediction',
      description='Repository developed for graduate research at Montclair '
                  'State University as part of the Applied Linguistics '
                  'program. The main idea is to use video game reviews '
                  'coupled with the amount of hours the reviewers played '
                  '(available through the Steam website) to build models of '
                  'reviewer experience. The original inspiration developed '
                  'during a deception detection class, the connection being '
                  'that experience is related to the truth: If a reviewer '
                  'truly lacks experience of a game, then their review '
                  '(whether it is intended to deceive or not) might betray '
                  'signs of that lack of experience.',
      long_description=readme(),
      version='0.1',
      author='Matt Mulholland et al.',
      author_email='mulhollandm2@lemur.montclair.edu',
      packages=find_packages(),
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      keywords='steam review video game',
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: ',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4'],
      zip_safe=False)

# Copy files from build/libs* directory (or try to guess where they are)
build_libs = [join(build_dir,
                   _dir)
              for _dir in listdir(build_dir) if _dir.startswith('lib')]
if not build_libs:
    stderr.write('Could not find "build" directory...\n')

if len(build_libs) == 1:
    build_lib = build_libs[0]
elif len(build_libs) > 1:
    build_lib = build_libs[0]
    stderr.write('Found multiple directories in {} that begin with "lib". '
                 'Trying first one: {}\n'.format(build_dir,
                                                 build_lib))
else:
    build_lib = None
    stderr.write('Found no directories in {} that begin with "lib".\n'
                 .format(build_dir))

if not build_lib:
    stderr.write('Could not find build/libs* directory. Checking to see if '
                 'the shared object files were generated in the project '
                 'directory or the current working directory.\n')
    for _dir in set([dirname(src_dir),
                     getcwd()]):
        stderr.write('Checking in {}...\n'.format(_dir))
        exts = [f for f in listdir(_dir) if f.endswith('.so')]
        stderr.write('Contents of {}:\n{}\n'.format(_dir,
                                                    '\n'.join(exts)))
        for ext in exts:
            if ext.startswith('features'):
                copy(join(_dir,
                          ext),
                     join(src_dir,
                          '{}.so'.format(ext.split('.', 1)[0])))
            else:
                copy(join(_dir,
                          ext),
                     join(util_dir,
                          '{}.so'.format(ext.split('.', 1)[0])))
else:
    exts = [f for f in listdir(build_lib) if f.endswith('.so')]
    for ext in exts:
        if ext.startswith('features'):
            copy(join(build_lib,
                      ext),
                 join(src_dir,
                      '{}.so'.format(ext.split('.', 1)[0])))
        else:
            copy(join(build_lib,
                      ext),
                 join(util_dir,
                      '{}.so'.format(ext.split('.', 1)[0])))

stderr.write('Complete.\n')
