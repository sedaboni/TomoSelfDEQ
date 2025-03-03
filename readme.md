# TomoSelfDEQ: Self-Supervised Deep Equilibrium Learning for Sparse-Angle CT Reconstruction

### [Paper (arXiv)](https://arxiv.org/abs/2502.21320) 

## Abstract
Deep learning has emerged as a powerful tool for solving inverse problems in imaging, including computed tomography (CT). However, most approaches require paired training data with ground truth images, which can be difficult to obtain, e.g., in medical applications. We present TomoSelfDEQ, a self-supervised Deep Equilibrium (DEQ) framework for sparse-angle CT reconstruction that trains directly on undersampled measurements. We establish theoretical guarantees showing that, under suitable assumptions, our self-supervised updates match those of fully-supervised training with a loss including the (possibly non-unitary) forward operator like the CT forward map. Numerical experiments on sparse-angle CT data confirm this finding, also demonstrating that TomoSelfDEQ outperforms existing self-supervised methods, achieving state-of-the-art results with as few as 16 projection angles.

![](/fig/Fig_16_angles.png)

## Setup Environment
1. Create the conda environment
```
conda env create --file "environment.yml"
```

2. Activate the environment
```
conda activate tomoselfdeq
```

3. Install LION
```
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive
pip install .
cd ..
```

## Run TomoSelfDEQ
```
python main.py --N_epochs=2000 --nangles=64 --loss_type=unsup --comment=test 
```
The results can be found  in the folder ./runs. 
Full training and testing logs/results can be found in this folder and loaded using tensorboard. 

## Credits
This project is partly based on the work of other researchers and developers:

- The dataset was adapted from [Sparse2Inverse](https://github.com/Nadja1611/Sparse2Inverse-Self-supervised-inversion-of-sparse-view-CT-data) by Nadja Gruber et al.

- The optimizer was sourced from [ScheduleFree](https://github.com/facebookresearch/schedule_free) developed by Meta Research.

# Citation
Please consider citing TomoSelfSEQ if you find it helpful.

```BibTex
@article{bubba2025tomoselfdeq,
  title={TomoSelfDEQ: Self-Supervised Deep Equilibrium Learning for Sparse-Angle CT Reconstruction},
  author={Bubba, Tatiana A and Santacesaria, Matteo and Sebastiani, Andrea},
  journal={arXiv preprint arXiv:2502.21320},
  year={2025}
}
 ```