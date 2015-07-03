from sys import (exit,
                 stderr)
from os import (listdir,
                getcwd)
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

# Copy files from build/libs* directory (or try to guess where they are)
build_libs_dirs_list = []
if exists(build_dir):
    build_libs_dirs_list = [join(build_dir,
                                 _dir) for _dir in listdir(build_dir)
                            if _dir.startswith('lib')]
    if not len(build_libs_dirs_list):
        stderr.write('Could not find "build" directory...\n')

if len(build_libs_dirs_list) == 1:
    build_libs_dir = build_libs_dirs_list[0]
elif len(build_libs_dirs_list) > 1:
    stderr.write('Found multiple directories in {} that begin with "lib". Not'
                 ' sure which one to use.\n'.format(build_dir))
else:
    stderr.write('Found no directories in {} that begin with "lib".\n'
                 .format(build_dir))
    build_libs_dir = None

if not exists(build_dir) or not build_libs_dir:
    stderr.write('Could not find build/libs* directory. Checking to see if '
                 'the shared object files were generated in the project '
                 'directory or the current working directory.\n')
    for _dir in [dirname(src_dir),
                 getcwd()]:
        stderr.write('Checking in {}...\n'.format(_dir))
        exts = [f for f in listdir(_dir) if f.endswith('.so')]
        stderr.write('Contents of {}:\n{}\n'.format(_dir,
                                                    '\n'.join(exts)))
        for ext in exts:
            if ext.startswith('feature_extraction'):
                copy(join(_dir,
                          ext),
                     join(src_dir,
                          '{}.so'.format(ext.split('.',
                                                   1)[0])))
            else:
                copy(join(_dir,
                          ext),
                     join(util_dir,
                          '{}.so'.format(ext.split('.',
                                                   1)[0])))
else:
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

stderr.write('Complete.\n')