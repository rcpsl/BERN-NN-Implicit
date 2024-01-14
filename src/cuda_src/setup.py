from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='bern_cuda_extensions',
      ext_modules=[
          CUDAExtension(name='ibf_minmax_cpp',
                        sources=['ibf_minmax/ibf_minmax.cpp', 'ibf_minmax/ibf_minmax_kernel.cu'],
                        extra_compile_args={'nvcc': ['-O2']}),
          CUDAExtension(name='row_wise_conv_cpp',
                        sources=['row_wise_conv/row_wise_conv.cpp', 'row_wise_conv/row_wise_conv_kernel.cu'],
                        extra_compile_args={'nvcc': ['-O2']}),
          CUDAExtension(name='ibf_dense_cpp',
                        sources=['ibf_to_dense/ibf_dense.cpp', 'ibf_to_dense/ibf_dense_kernel.cu'],
                        extra_compile_args={'nvcc': ['-O2']}),
          CUDAExtension(name='find_indices_cpp',
                        sources=['find_indices/find_indices.cpp', 'find_indices/find_indices_kernel.cu'],
                        extra_compile_args={'nvcc': ['-O2']}),
      ],
      cmdclass={'build_ext': BuildExtension})
