from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="SynchrotronSelfCompton", sources=["SynchrotronSelfCompton.pyx"])
setup(ext_modules=cythonize(ext))
'''

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("SynchrotronSelfCompton",
              ["SynchrotronSelfCompton.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
  name = "thread_demo",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)

'''
