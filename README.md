# Language is all omics needs: Unifying single-cell omics through language with CellHermes
## Introduction 
This repository hosts the official implementation of CellHermes, a framework that can unify heterogeneous single-cell omics data by existing LLMs. Upon powerful capabilitz of LLMs, such as text-based understanding and reasoning, we can used it as encoder, predictor and explainer. This design allows CellHermes to span the entire research loop from representation learning to prediction and interpretability.
<p align="center"><img src="https://github.com/bm2-lab/CellHermes/blob/main/img/Framework.png" alt="CellHermes" width="900px" /></p> 

## Installation
Our experiments were conducted on python=3.10.15 and our CUDA version is 12.6.
We recommend using Anaconda / Miniconda to create a conda environment for using CellHermes. You can create a python environment using the following command:
```python
conda create -n CellHermes python==3.10.15
```

Then, you can activate the environment using:
```python
conda activate CellHermes
```


### Pretrained Checkpoint
We release these variants of ​​CellHermes​​. Please download to the `pretrained_ckpt` directory.
| Model Name | Stage | Description |
|------------|-----------| -------|
| [CellHermes](https://huggingface.co/)   | Pretraining | Spiking LLMs model on single-cell transcriptomic data and PPI network, simultaneously |
| [CellHermes-Multi-Task](https://huggingface.co/) | Instruction fine-tuning |   |
| [CellHermes-T-Cell-Reactivity](https://huggingface.co/) | Instruction fine-tuning |   |

## Citation  
## Contacts
bm2-lab@tongji.edu.cn  
gao.yicheng.98@gmail.com
