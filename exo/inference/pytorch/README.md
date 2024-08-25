# PyTorch & HuggingFace inference engine
Experimental, still under development


## Install
Install needed py modules, make sure to be using CUDA 12.4 for the PyTorch install

```console
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
$ pip install transformers accelerate
```

After installing accelerate you get hit with a dependency error, for now ignore until we can fix this as exo works fine with 1.26.4

```console
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
exo 0.0.1 requires numpy==2.0.0, but you have numpy 1.26.4 which is incompatible.
```