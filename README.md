# Protein pocket detection
GitHub repository for the paper: Protein ligand binding site prediction using graph transformer neural network, by Ryuichiro Ishitani, Mizuki Takemoto, and Kentaro Tomii.

## Environment
This package run under the environment:
* Python: 3.10
* PyTorch: 2.0.1
* PyG: 

The other package dependencies are described in [requirements.txt](/requirements.txt).

To install the package and dependencies:
```
pip install .
```

## Usage
### Download dataset and model weights
If you want to use the pretrained model described in the paper,
please download the dataset and model weights files from zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6984632.svg)](https://doi.org/10.5281/zenodo.6984632)
and save to the [data](/data) directory.


### Inference using PDB file
To train the policy using the specific reward functions, run the script contained in the specific subdirectories.
* [Penalized LogP](/examples/penalized_logp)
* [Similarity](/examples/similarity)

## Citation
If you find our work relevant to your research, please cite:
```
@article{ishitani2022rjtrl,
    title={Molecular design method using a reversible tree representation of chemical compounds and deep reinforcement learning},
    author={Ryuichiro Ishitani and Toshiki Kataoka and Kentaro Rikimaru},
    year={2022},
    journal={J. Chem. Inf. Model. https://doi.org/10.1021/acs.jcim.2c00366}
}
```
