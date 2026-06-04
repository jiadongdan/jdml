# Jupyter Kernel Dies With PyTorch and Matplotlib

## Summary

A Jupyter kernel can crash when PyTorch and matplotlib, NumPy, SciPy, scikit-learn, or other scientific Python packages are used together on Windows. A common cause is an OpenMP runtime conflict, where more than one OpenMP DLL is loaded into the same Python process.

This is usually an environment issue, not a bug in model code or plotting code.

The typical error is:

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

In a terminal, this may terminate the script. In Jupyter, it can kill the whole kernel.

## Why It Happens

PyTorch and scientific Python packages often depend on compiled native libraries for fast CPU computation. These libraries may use OpenMP for parallel execution.

On Windows, duplicate or incompatible OpenMP runtimes can be loaded from different package locations, for example:

```text
...\site-packages\torch\lib\libiomp5md.dll
...\envs\<env-name>\Library\bin\libiomp5md.dll
```

Even if the DLL file name is the same, loading two physical copies into one Python process can trigger Intel OpenMP's duplicate-runtime check.

This can happen more easily in notebooks because a Jupyter kernel is long-running: once one package loads a native runtime, later imports may try to load another one into the same process.

## Quick Notebook Workaround

Restart the kernel and run this as the first cell, before importing `torch`, `numpy`, `matplotlib`, `sklearn`, or project modules that import them:

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

Then continue with normal imports:

```python
import torch
import matplotlib.pyplot as plt
```

This workaround is often acceptable for local notebook experimentation, but it is not the cleanest long-term fix.

## Prefer Lightweight Imports

If a task does not need PyTorch, avoid importing model modules that load PyTorch.

For example, plotting a CNN architecture from a config dictionary can be done without constructing the actual PyTorch model:

```python
from jdml.visualization import plot_cnn_architecture

plot_cnn_architecture(config)
```

This keeps the notebook lighter and reduces the chance of triggering native-library conflicts.

## How To Inspect Loaded DLLs

Inside Jupyter, after importing the packages you want to inspect:

```python
import os
import psutil

process = psutil.Process(os.getpid())
keywords = ["iomp", "omp", "mkl", "openblas", "vcomp", "tbb"]

dlls = []
for module in process.memory_maps():
    path = module.path
    if any(keyword in path.lower() for keyword in keywords):
        dlls.append(path)

for path in sorted(set(dlls)):
    print(path)
```

Suspicious signs include duplicate OpenMP runtimes from different locations, especially multiple copies of:

```text
libiomp5md.dll
```

Other relevant native libraries may include:

```text
mkl_core.dll
mkl_intel_thread.dll
libomp.dll
vcomp140.dll
openblas.dll
tbb.dll
```

## Longer-Term Fix

Create a clean Conda environment with packages installed from consistent channels. The goal is to let Conda resolve a coherent binary dependency stack, so PyTorch, NumPy, matplotlib, MKL/OpenMP, and related packages do not load duplicate or incompatible native runtimes.

Example for a CUDA-enabled PyTorch environment:

```powershell
conda create -n torch-clean python=3.11
conda activate torch-clean
conda install -c pytorch -c nvidia -c defaults pytorch torchvision torchaudio pytorch-cuda=12.1
conda install -c defaults numpy matplotlib scipy scikit-learn ipykernel
python -m ipykernel install --user --name torch-clean --display-name "Python (torch-clean)"
```

For a CPU-only environment, omit the CUDA package:

```powershell
conda create -n torch-clean python=3.11
conda activate torch-clean
conda install -c pytorch -c defaults pytorch torchvision torchaudio cpuonly
conda install -c defaults numpy matplotlib scipy scikit-learn ipykernel
python -m ipykernel install --user --name torch-clean --display-name "Python (torch-clean)"
```

After creating the new environment, rerun the DLL inspection code in Jupyter and check whether duplicate OpenMP runtimes are still loaded.
