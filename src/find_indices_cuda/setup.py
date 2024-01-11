from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='find_indices_extension',
    ext_modules=[
        CUDAExtension('find_indices_extension', [
            'find_indices_wrapper.cpp',
            'find_indices_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
