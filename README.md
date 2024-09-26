# BERN-NN-IBF 
Bounding neural networks with Bernstein polynomials in Implicit Form

## Directory Layout

Most of our experiment code is in the `rep/` directory.
Some parts of the implementation are currently in `src/`, The cuda extensions
are in `src/cuda_src`.

## Environment Setup

### 1. Create a Conda environment. 

The code uses pytorch extensions written with CUDA, so
it requires an NVIDIA GPU. There is not a CPU-only version. 
We have the version set to CUDA-11.7, but the versions may be changed to match your hardware.

```console
conda env create -n bern-nn -f conda/pytorch-2.0.1-cuda-11.7.yaml
conda activate bern-nn
```

### 2. Build the pytorch extensions

This step should be done while inside the newly created conda environment.
This requires a CUDA compiler, like nvcc. 

```console
pip install -e .
```

This pip install build compile and install the PyTorch CUDA extensions specified in `setup.py`.
This install can be be slow. By default, we have the flag `-O2` enabled. You may wish to specify 
a particular architecture. This can be done by setting `export TORCH_CUDA_VERSION=<version>` 
before running the pip install.

## Running the example

We provide a self-contained example showing how to use the `NetworkModule` class in `rep/Bern_NN_IBF.py`.
Comments in this example script explain the different settings that can be passed into the `NetworkModule`.

```console
python example.py
```
