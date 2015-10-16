from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
#import cython_gsl
import numpy as np

extNames = ['tilecode'] 

def makeExtension(extName):
    extPath = extName + ".pyx"
    return Extension(
        extName,
        [extPath],
        libraries=["m",],
        #libraries=cython_gsl.get_libraries(), #.append("m"),
        #library_dirs=[cython_gsl.get_library_dir()],
        include_dirs = [np.get_include(),"..", "../include"], #, cython_gsl.get_cython_include_dir()],
        extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"], #, "-std=c++11"
        extra_link_args=['-fopenmp'],
        )

extensions = [makeExtension(name) for name in extNames]

setup(
  name = 'econlearn',
  packages =["econlearn"],
  #include_dirs = [cython_gsl.get_include()],
  cmdclass = {'build_ext': build_ext}, 
  ext_modules = extensions, 
) 
