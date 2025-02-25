# llm-local-forge
在个人PC上体验LLM预训练完整流程。

# 环境准备
```shell
conda create -n llm-local-forge python=3.11

conda activate llm-local-forge
```

```shell
pip install poetry
poetry init
poetry add torch torchvision torchaudio
poetry add transformers
poetry add datasets
```
