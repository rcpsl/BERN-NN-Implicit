from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='ibf_dense_cpp',
      ext_modules=[
          CUDAExtension('ibf_dense_cpp', ['ibf_dense.cpp', 'ibf_dense_kernel.cu'])
      ],
      cmdclass={'build_ext': BuildExtension})

