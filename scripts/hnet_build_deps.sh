#!/bin/bash
# Builds the H-Net CUDA dependencies (causal_conv1d, mamba_ssm (Triton-only), flash_attn) into .mt_env
# for torch 2.14+cu126, targeting sm_89 (L40S). Sources + patches live in .build/src, and the
# CUDA 12.6 toolkit is NVIDIA's redist nvcc archive plus headers/libs from pip wheels (.build/cudapkgs) because the system nvcc is 12.2 and
# does not support gcc 13.
# Patches already applied in .build/src: mamba_ssm/__init__ tolerates the missing Mamba-1 CUDA ext,
# causal-conv1d builds only sm_89, flash-attn gets an sm_89 target and -std=c++20 (torch 2.14 headers need C++20).
# Run on the LOGIN node (bash scripts/hnet_build_deps.sh): the compilation fails on the GPU compute nodes.
set -e
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
B=$ROOT/.build
export PYTHONPATH=$B/tools:$PYTHONPATH PATH=$B/tools/bin:$PATH
# (pip, ninja, packaging, wheel, psutil were vendored into .build/tools on the login node)

# Assemble a CUDA_HOME from the pip wheels.
C=$B/cuda126
if [ ! -x $C/bin/nvcc ]; then
  rm -rf $C; mkdir -p $C/include $C/lib64
  cp -r $B/cuda_nvcc-linux-x86_64-12.6.85-archive/. $C/  # NVIDIA redist nvcc (the pip wheel has only ptxas)
  for d in $B/cudapkgs/nvidia/*/include; do cp -rn $d/. $C/include/; done
  for d in $B/cudapkgs/nvidia/*/lib; do cp -rn $d/. $C/lib64/; done
  (cd $C/lib64 && for f in *.so.[0-9]*; do ln -sf $f ${f%%.so.*}.so; done)
  chmod +x $C/bin/* $C/nvvm/bin/* 2>/dev/null || true
fi
export CUDA_HOME=$C PATH=$C/bin:$PATH LIBRARY_PATH=$C/lib64:$LIBRARY_PATH
nvcc --version | tail -2
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 NVCC_THREADS=2
export PATH=$C/bin:$PATH
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=89
cd $B/src
for pkg in causal-conv1d mamba flash_attn-2.8.3; do
  echo "=== building $pkg ($(date))"
  nice -n 10 python3 -m pip install -v --no-build-isolation --no-deps --upgrade --target $ROOT/.mt_env/lib ./$pkg 2>&1 | grep -vE "^\s*(copying|creating|adding)" | tail -40
done
echo "=== done ($(date))"
python3 -m pip install -q --target $ROOT/.mt_env/lib huggingface_hub einops optree omegaconf matplotlib
cd $ROOT
python3 -c "import causal_conv1d_cuda, flash_attn_2_cuda, mamba_ssm; from mamba_ssm.modules.mamba2 import Mamba2; import flash_attn; print('imports ok', flash_attn.__version__)"
