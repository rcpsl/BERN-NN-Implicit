from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='generate_binom_coeffs_extension',
    ext_modules=[
        CUDAExtension('generate_binom_coeffs_extension', [
            'generate_binom_coeffs_wrapper.cpp',
            'generate_binom_coeffs_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
