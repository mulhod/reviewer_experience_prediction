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
from setuptools import setup
from distutils.extension import Extension

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = True

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON=='auto':
            USE_CYTHON=False
        else:
            raise

cmdclass = {}

def readme():
    with open('README.md') as f:
        return f.read()

# Hackish way of doing this. Find better way...
root_env = getoutput('conda info | grep "package cache :"'
                     ' | awk \'{print $4}\'')
# Try to guess the location of the conda installation
if not root_env:
    root_env = '/home/{}/conda'.format(getuser())
python_header_dir = join(root_env,
                         '.pkgs/python-3.4.3-0/include/python3.4m')

if USE_CYTHON:
    ext_modules = [Extension('src.features',
                             ['src/features.pyx'],
                             include_dirs=[python_header_dir]),
                   Extension('util.datasets',
                             ['util/datasets.pyx'],
                             include_dirs=[python_header_dir]),
                   Extension('util.mongodb',
                             ['util/mongodb.pyx'],
                             include_dirs=[python_header_dir])]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = [Extension('src.features',
                             ['src/features.c'],
                             include_dirs=[python_header_dir]),
                   Extension('util.datasets',
                             ['util/datasets.c'],
                             include_dirs=[python_header_dir]),
                   Extension('util.mongodb',
                             ['util/mongodb.c'],
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
      url='https://github.com/mulhod/reviewer_experience_prediction',
      long_description=readme(),
      version='0.1',
      author='Matt Mulholland et al.',
      author_email='mulhollandm2@montclair.edu',
      packages=['data', 'src', 'util'],
      package_data={'data': ['*.jsonlines']},
      include_package_data=True,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      entry_points={'console_scripts':
                    ['extract_features = src.extract_features:main',
                     'learn = src.learn:main',
                     'get_review_data = util.get_review_data:main',
                     'make_arff_files = util.make_arff_files:main',
                     'make_train_test_sets = '
                     'util.make_train_test_sets:main']},
      keywords='steam review video game',
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: ',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS'],
      zip_safe=False)
