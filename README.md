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

Then, you can activate the environment using and install the required packages:
```python
conda activate CellHermes
pip install -r requirement.txt
```
The training of CellHermes was performed by LLaMA-Factory (version: 0.9.1), so it is needed to configure the environment according to the environment of LLaMA-Factory. This can be done by switching to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) or directly use the following command:
```
cd LLama_factory_v0.9.1.dev0 
pip install -e ".[torch,metrics]"
```

### 🤖 Pretrained Checkpoint
We release these variants of ​​CellHermes​​. Please download to the `model_ckpt` directory.
| Model Name | Stage | Description |
|------------|-----------| -------|
| [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   | Base | The base LLM used in this study |
| [CellHermes](https://huggingface.co/)   | Pretraining | Spiking LLMs model on single-cell transcriptomic data and PPI network, simultaneously |
| [CellHermes-Multi-Task](https://huggingface.co/) | Instruction fine-tuning | Instruction-tuned model adapter with 7 databases across 10 tasks |
| [CellHermes-T-Cell-Reactivity](https://huggingface.co/) | Instruction fine-tuning | Instruction-tuned model adapter with T cell tumor-reactivity prediction task  |

### 🚀 Training

Model training is conducted on 2 NVIDIA RTX A6000 GPUs.

```bash
conda activate CellHermes
cd LLama_factory_v0.9.1.dev0 
bash ../bash_config/pretrain.sh
llamafactory-cli export ../bash_config/merge_lora_config.yaml
```
### 🔆 As an encoder
The following are commands for encoding biological entities by CellHermes, such as genes, cells and cell-specific genes.
#### Obtaining gene embeddding for a given gene
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "Gene BRCA1" \
                    -o "./saves/gene_tmp_emb.pkl"
```
#### Obtaining cell embeddding for a given cell transcriptomic information (gene rank in this case)
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "A cell with genes ranked by expression: MALAT1 TMSB4X B2M SRGN FTH1 BTG1 GNLY TPT1 EEF1A1 HLA-A ZFP36L2 PTMA HLA-B TMSB10 XCL1 PABPC1 ANXA1" \
                    -o "./saves/cell_tmp_emb.pkl"
```
#### Obtaining gene embedding for a given gene from a specific cell with its transcriptomic information (gene rank in this case)
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "A cell with genes ranked by expression: MALAT1 TMSB4X B2M RGS1 CCL3 CCL4 CD69 JUNB HSP90AA1 ZFP36 FTH1 DNAJB1 DUSP1 SAT1 CXCR4. In this cell, Gene BRCA1" \
                    -o "./saves/cell_specific_gene_tmp_emb.pkl"
```
### 🔆 As a predictor
The following are commands for fine-tuning CellHermes with multiple task datasets, such as perturbation prediction, cell fitness prediction, gene interaction prediction, etc. Users can change the `--dataset` parameter in `multitask_ft.sh` file to incorporate any dataset they want.
```bash
conda activate CellHermes
cd LLama_factory_v0.9.1.dev0 
bash ../bash_config/multitask_ft.sh
```
### 🔆 As an explainer

### 🌻 Acknowledgement
We gratefully acknowledge the use some of codes from the following projects: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [scGPT](https://github.com/bowang-lab/scGPT), [GenePT](https://github.com/yiqunchen/GenePT). Our work builds upon their foundational contributions.

## 🔖 Citation  
## 😀 Contacts
bm2-lab@tongji.edu.cn  
gao.yicheng.98@gmail.com
