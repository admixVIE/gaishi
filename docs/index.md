# GAISHI

`GAISHI` is a Python package for **G**enomic **A**nalysis of **I**ntrogressed-**S**ite and -**H**aplotype **I**dentification using machine learning.

It supports:

- logistic regression models
- extra-trees classifiers
- UNet++ models

### Requirements

`GAISHI` works on Unix/Linux operating systems and is tested with the environment in this [env.yaml](https://github.com/xin-huang/gaishi/blob/main/env.yaml):

- Python 3.9.19
- Python packages:
    - black=24.10.0
    - demes=0.2.3
    - flake8=7.1.1
    - h5py=3.10.0
    - msprime=1.3.1
    - numpy=1.26.4
    - onnx=1.11.0
    - onnxconverter-common==1.9.0
    - onnxruntime=1.19.2
    - ortools==8.2.8710
    - pandas=2.2.1
    - pip=24.0
    - protobuf=3.20.0
    - pydantic=2.11.7
    - pyranges=0.0.129
    - pytest=8.1.1
    - pytest-cov=5.0.0
    - pyyaml=6.0.1
    - safetensors==0.4.5
    - scikit-allel=1.3.7
    - scikit-learn=1.4.1.post1
    - scipy=1.12.0
    - skl2onnx==1.11.2
    - torch==2.2.0
    - tskit=0.5.6
    - https://github.com/xin-huang/seriate.git@dev

### Installation

Users can install `GAISHI` with [mamba](https://github.com/mamba-org/mamba):

```
git clone https://github.com/xin-huang/gaishi
cd gaishi
mamba env create -f build-env.yaml
conda activate gaishi
pip install .
```

To check the installation:

```
gaishi -h
```

### Citations

If you find `GAISHI` useful, please cite:

- Huang X, Hackl J, Pawar H, Kuhlwilm M. 2026. GAISHI: A Python package for detecting ghost introgression with machine learning. bioRxiv: 2026.01.31.703038. 
