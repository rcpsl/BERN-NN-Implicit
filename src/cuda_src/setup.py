from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

flags = ['-O2']

setup(name='bern_cuda_extensions',
    ext_modules=[
          CUDAExtension(name='ibf_minmax_cpp',
                        sources=[
                            'ibf_minmax/ibf_minmax.cpp',
                            'ibf_minmax/ibf_minmax_kernel.cu',
                            'ibf_minmax/quadrant_ibf_minmax_kernel.cu'
                        ],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='row_wise_conv_cpp',
                        sources=['row_wise_conv/row_wise_conv.cpp', 'row_wise_conv/row_wise_conv_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='ibf_dense_cpp',
                        sources=['ibf_to_dense/ibf_dense.cpp', 'ibf_to_dense/ibf_dense_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='find_indices_cpp',
                        sources=['find_indices/find_indices.cpp', 'find_indices/find_indices_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='generate_binom_coeffs_cpp',
                        sources=['generate_binom_coeffs/generate_binom_coeffs.cpp',
                                 'generate_binom_coeffs/generate_binom_coeffs_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='repeat_terms_cpp',
                        sources=['repeat_terms/repeat_terms.cpp', 'repeat_terms/repeat_terms_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
          CUDAExtension(name='repeat_terms_2_cpp',
                        sources=['repeat_terms_2/repeat_terms_2.cpp', 'repeat_terms_2/repeat_terms_2_kernel.cu'],
                        extra_compile_args={'nvcc': flags}),
      ],
      cmdclass={'build_ext': BuildExtension})
