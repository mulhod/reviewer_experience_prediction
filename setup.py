from sys import (exit,
                 stderr,
                 version_info)
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
from distutils.command.build import build as _build

if version_info < (3, 0):
    raise Exception('Installation requires Python >= 3.0.')

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = True

cmdclass = {}

class build(_build):
    def run(self):
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print(numpy.get_include())
        _build.run(self)


def readme():
    with open('README.md') as f:
        return f.read()


def reqs():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def get_python_header_dir():
    # Hackish way of doing this. Find better way...
    pkgs_dir = getoutput('conda info | grep "package cache :"'
                         ' | awk \'{print $4}\'')
    if not exists(pkgs_dir):
        raise ValueError('.pkgs folder could not be located via "conda info"')
    pkgs_dir = join(pkgs_dir, '.pkgs')
    python_dirs = [_dir for _dir in listdir(pkgs_dir) if _dir.startswith('python-3.')]
    if len(python_dirs) == 1:
        pkgs_dir = join(pkgs_dir, python_dirs[0], 'lib')
    else:
        raise ValueError('Multiple "python-3.*" directories found in {0}'
                         .format(pkgs_dir))
    python_dirs = [_dir for _dir in listdir(pkgs_dir) if _dir.startswith('python3')]
    if len(python_dirs) == 1:
        pkgs_dir = join(pkgs_dir, python_dirs[0])
    else:
        raise ValueError('Multiple "python3.*" directories found in {0}'
                         .format(pkgs_dir))
    return pkgs_dir


def get_ext_modules(use_cython=USE_CYTHON):
    global cmdclass
    python_header_dir = get_python_header_dir()

    if use_cython:
        try:
            from Cython.Distutils import build_ext
        except ImportError:
            stderr.write('\"cython\" is not installed. Forcing \"USE_CYTHON\" '
                         'to \"False\" in order to install.\n')
            use_cython = False

    if use_cython:
        ext_modules = [Extension('src.features',
                                 ['src/features.pyx'],
                                 include_dirs=[python_header_dir]),
                       Extension('src.experiments',
                                 ['src/experiments.pyx'],
                                 include_dirs=[python_header_dir]),
                       Extension('src.datasets',
                                 ['src/datasets.pyx'],
                                 include_dirs=[python_header_dir]),
                       Extension('src.mongodb',
                                 ['src/mongodb.pyx'],
                                 include_dirs=[python_header_dir])]
        cmdclass.update({'build_ext': build_ext})
        return ext_modules

    ext_modules = [Extension('src.features',
                             ['src/features.c'],
                             include_dirs=[python_header_dir]),
                   Extension('src.experiments',
                             ['src/experiments.pyx'],
                             include_dirs=[python_header_dir]),
                   Extension('src.datasets',
                             ['src/datasets.c'],
                             include_dirs=[python_header_dir]),
                   Extension('src.mongodb',
                             ['src/mongodb.c'],
                             include_dirs=[python_header_dir])]
    return ext_modules


setup(name='Reviewer Experience Prediction',
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
      author_email='mulhodm@gmail.com',
      setup_requires=['numpy'],
      install_requires=reqs(),
      packages=['data', 'src', 'util', 'tests'],
      package_data={'data': ['*.jsonlines']},
      include_package_data=True,
      ext_modules=get_ext_modules(),
      cmdclass=cmdclass,
      scripts=['setup.sh', 'util/get_review_data_all_games.sh'],
      entry_points={
          'console_scripts':
              ['extract_features = util.extract_features:main',
               'cv_learn = util.cv_learn:main',
               'get_review_data = util.get_review_data:main',
               'make_arff_files = util.make_arff_files:main',
               'make_train_test_sets = util.make_train_test_sets:main',
               'create_mongodb_index = util.create_mongodb_index:main']},
      keywords='nlp machine learning video game review steam',
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS'],
      test_suite='nose2.collector.collector',
      zip_safe=False)
