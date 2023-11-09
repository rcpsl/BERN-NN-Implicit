from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='row_wise_convolution_wrapper',
      ext_modules=[
          CUDAExtension('row_wise_convolution_wrapper', ['row_wise_convolution_wrapper.cpp', 'row_wise_convolution_kernel.cu'])
      ],
      cmdclass={'build_ext': BuildExtension})

