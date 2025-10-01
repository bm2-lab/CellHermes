# Language is all omics needs: Unifying single-cell omics through language with CellHermes
## 💡 Introduction 
This repository hosts the official implementation of CellHermes, a framework that can unify heterogeneous single-cell omics data by existing LLMs. Upon powerful capabilitz of LLMs, such as text-based understanding and reasoning, we can used it as encoder, predictor and explainer. This design allows CellHermes to span the entire research loop from representation learning to prediction and interpretability.
<p align="center"><img src="https://github.com/bm2-lab/CellHermes/blob/main/img/Framework.png" alt="CellHermes" width="900px" /></p> 

## 🔧 Environment
Our experiments were conducted on python=3.10.15 and our CUDA version is 12.6.
We recommend using Anaconda / Miniconda to create a conda environment for using CellHermes. You can create a python environment using the following command:
```python
conda create -n CellHermes python==3.10.15
```

Then, you can activate the environment using:
```python
conda activate CellHermes
```
The training of CellHermes was performed by LLaMA-Factory (version: 0.9.1), so it is needed to configure the environment according to the environment of LLaMA-Factory. This can be done by switching to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) or directly use the following command:
```
cd LLama_factory_v0.9.1.dev0 
pip install -e ".[torch,metrics]"
```

### 🤖 Pretrained Checkpoint
We release these variants of ​​CellHermes​​. Please download to the `pretrained_ckpt` directory.
| Model Name | Stage | Description |
|------------|-----------| -------|
| [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   | Base | The base LLM used in this study |
| [CellHermes](https://huggingface.co/)   | Pretraining | Spiking LLMs model on single-cell transcriptomic data and PPI network, simultaneously |
| [CellHermes-Multi-Task](https://huggingface.co/) | Instruction fine-tuning |   |
| [CellHermes-T-Cell-Reactivity](https://huggingface.co/) | Instruction fine-tuning |   |

## 🔖 Citation  
## 😀 Contacts
bm2-lab@tongji.edu.cn  
gao.yicheng.98@gmail.com
