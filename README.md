This repo contains code for the paper "Don’t Cut Corners: A theory of disentanglement in biologically inspired neurons for (almost) any dataset". 

The following files can be used to generate each of the figures:
1. Figure 1c: Fig1_Basic.ipynb
2. Figure 1d: 'Figure 1d.ipynb'
3. Figure 2 Left: LUKE
4. Figure 2 Right: KYLE
5. Figure 3a-d: RNN_linear_modularity_torch.py & RNN_nonlinear_modularity.py.
6. Figre 3e: see 'Actionable Neural Representations: Grid Cells from Minimal Constraints', 2023.
7. Figure 3f-g: 'Figure 3fg.ipynb'
8. Figure 4: WILL
9. Figure 5: Dual_Motor_RNNs.ipynb and Single_Motor_RNNs.ipynb
10. Figure 6: WILL
11. Figure 7: WILL


### Installation
```
mamba create -y -n modular python=3.12
mamba activate modular
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
