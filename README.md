### Installation
```
mamba create -y -n modular python=3.12
mamba activate modular
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

### Cluster
```
mamba create -n modular_cuda11 python=3.10 -y && mamba activate modular_cuda11

pip install "jaxlib==0.4.7+cuda11.cudnn82" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "jax==0.4.7"
pip install "numpy<2"
pip install nvidia-cudnn-cu11==8.6.0.163
pip install -r requirements_cuda11.txt


pip install "jaxlib==0.4.25+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "jax==0.4.25"
pip install nvidia-cudnn-cu11==8.6.0.163
pip install nvidia-cufft-cu11==10.9.0.58
pip install nvidia-cusolver-cu11==11.4.0.1
pip install nvidia-cuda-cupti-cu11==11.8.87
pip install nvidia-cusparse-cu11==11.7.5.86
pip install -r requirements.txt

```

### Add environment variables to `mamba activate`
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mamba deactivate && mamba activate modular
```