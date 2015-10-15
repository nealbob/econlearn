from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
#import cython_gsl
import numpy as np

ext_modules=[
    Extension("tilecode",
              ["tilecode.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"], #, "-std=c++11"
              include_dirs = [np.get_include(),"..", "../include"], #, cython_gsl.get_cython_include_dir()],
              extra_link_args=['-fopenmp'],
              ) 
]

setup(
  name = "tilecode",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)


