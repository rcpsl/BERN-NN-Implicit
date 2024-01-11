from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='repeat_terms_2_extension',
    ext_modules=[
        CUDAExtension('repeat_terms_2_extension', [
            'repeat_terms_2_wrapper.cpp',
            'repeat_terms_2_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
