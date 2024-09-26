# BERN-NN-IBF 
Bounding neural networks with Bernstein polynomials in Implicit Form

## Directory Layout

    .
    ├── rep                   # "Reproduction," primarily for experiment scripts.  
    ├── src                   # Source files that may be used in the experiment scripts. 
    ├── src/cuda_src          # Contains PyTorch extensions written in CUDA.
    ├── conda                 # Conda environment files. Used to install dependencies.
    ├── setup.py              # Build and install cuda extensions. 
    └── example.py		# A self-contained example for running the `NetworkModule`  
				# to compute lower bounds.



Most of our experiment code is in the `rep/` directory. Some parts of the implementation and experiments are currently in `src/`, The cuda extensions
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

After the conda environment and cuda dependencies are setup, you can run our self-contained example. 
This illustrates simple usage of the `NetworkModule` class in `rep/Bern_NN_IBF.py`. Comments in this 
example script explain some of the different settings that can be passed into the `NetworkModule`.

```console
python example.py
```
