This repo contains code for the paper "Don’t Cut Corners: A theory of disentanglement in biologically inspired neurons for (almost) any dataset". 

The following notebooks can be used to generate each of the figures:
1. Figure 1:
2. Figure 2:
3. Figure 3:
4. Figure 4:
5. Figure 5: Dual_Motor_RNNs.ipynb and Single_Motor_RNNs.ipynb
6. Figure 6: 


### Installation
```
mamba create -y -n modular python=3.12
mamba activate modular
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
