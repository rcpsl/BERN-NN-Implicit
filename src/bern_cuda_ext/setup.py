from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='ibf_minmax_cpp',
      ext_modules=[
          CUDAExtension('ibf_minmax_cpp', ['ibf_minmax.cpp', 'ibf_minmax_kernel.cu'])
      ],
      cmdclass={'build_ext': BuildExtension})

