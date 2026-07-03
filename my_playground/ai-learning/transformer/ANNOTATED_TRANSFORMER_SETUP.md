# Annotated Transformer Environment

The upstream Harvard NLP repo currently lists this dependency line in
`requirements.txt`:

```text
--find-links https://download.pytorch.org/whl/torch_stable.html pandas==1.3.5 torch==1.11.0+cu113 torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair==4.1 jupytext==1.13 flake8 black GPUtil wandb
```

Those pins are from the PyTorch 1.11 / Python 3.6-3.9 era and include a
CUDA-specific PyTorch wheel. On this machine, the repo `.venv` points at
Python 3.13, while compatible TorchText wheels for this path are available for
Python 3.12. Use a dedicated Python 3.12 environment for the notebook.

## Setup

From the repo root:

```bash
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv venv .venv-transformer --python 3.12
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv pip install \
  --python .venv-transformer/bin/python \
  -r my_playground/ai-learning/transformer/annotated_transformer_requirements.txt
.venv-transformer/bin/python -m ipykernel install \
  --user \
  --name annotated-transformer \
  --display-name "Python 3.12 (Annotated Transformer)"
```

Then open `my_playground/ai-learning/transformer/The Annotated Transformer.ipynb`
and choose the `Python 3.12 (Annotated Transformer)` kernel.

## Why These Versions

- `torch==2.3.0` and `torchtext==0.18.0` are the modern pair closest to the
  old notebook API while still having macOS arm64 / Python 3.12 wheels.
- `torchdata==0.8.0` is pinned because later TorchData releases removed the
  DataPipes APIs that older TorchText dataset helpers still expect.
- `pandas==1.3.5` and `spacy==3.2` from upstream are too old for this repo's
  Python target, so the setup uses current compatible major versions.
- The spaCy English and German model wheels are installed directly from their
  release URLs so the notebook can load `en_core_web_sm` and `de_core_news_sm`
  without separate download commands.
- The notebook's inline `!pip install torchtext==0.12` cells should be skipped;
  they are the environment issue.
