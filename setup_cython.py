"""Build Cython extensions for ICARUS performance-critical kernels."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "project_icarus.cython_kernels.atmosphere_cython",
        ["project_icarus/cython_kernels/atmosphere_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="icarus_cython_kernels",
    ext_modules=cythonize(extensions, compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
    zip_safe=False,
)
