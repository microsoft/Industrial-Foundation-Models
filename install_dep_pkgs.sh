#! /bin/bash

# 24/05/28 Note: CUDA 12.1 is the latest version supported by pytorch and flash-attn, using CUDA 12.4 does not trigger any error, but CUDA 12.5 will

# We install the latest pytorch version with cuda 12.1 since pytorch did not support cuda 12.4 until 24/05/28
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# Each time you upgrdae pytorch, you need to re-install flash-attn
pip install packaging
pip uninstall -y ninja && pip install ninja
# pip install flash-attn --no-build-isolation
# if the latest flash-attn does not work, you can install the previous version
pip install flash-attn==2.5.8 --no-build-isolation

# Install other dependencies, required by llama 2, hugging face
pip install accelerate
pip install appdirs
pip install loralib
# Starting from 0.43.1, bitsandbytes supports cuda 12.4
pip install bitsandbytes==0.43.1
# in case bitsandbytes cannot find triton, you need to re-install triton
# pip install triton==2.3.0
pip install black
pip install black[jupyter]
pip install datasets
pip install fire
# Starting from 4.41.1, transformers supports llama 3/2, phi 3/2, etc.
# pip install transformers==4.41.1
pip install transformers==4.43.1
pip install sentencepiece
pip install py7zr
pip install scipy
# pip install gradio
# pip install openai
pip install promptsource
pip install scikit-learn
pip install wandb
pip install kaggle
pip install hf_transfer

# For generating synthetic data
# pip install tabpfn
# pip install gpytorch
# pip install ConfigSpace
