from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointwise_div_extension',
    ext_modules=[
        CUDAExtension('pointwise_div_extension', [
            'pointwise_div_wrapper.cpp',
            'pointwise_div_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
