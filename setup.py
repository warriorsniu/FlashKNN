from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='FlashKNN',
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["FlashKNN"],
    package_dir={"FlashKNN": "functions"},
    ext_modules=[
        CUDAExtension('FlashKNN.CuFun', [
        'csrc/flash_knn_query_dynamic_load.cu',
        'csrc/flash_knn_query_global_memory.cu',
        'csrc/flash_knn_query_GMPS.cu',
        'csrc/flash_knn_query.cu',
        'csrc/api.cpp'
        ],
        extra_compile_args={'cxx': ['-g', "-O3", "-mavx2", "-funroll-loops"],
                            'nvcc': ['-O2', '-arch=sm_86', "-Xptxas", "-v", "-lineinfo"]})
    ],
    cmdclass={'build_ext': BuildExtension},
    include_dirs=["csrc/"]
)