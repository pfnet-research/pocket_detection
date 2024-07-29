# Protein pocket detection
GitHub repository for the paper: Protein ligand binding site prediction using graph transformer neural network, by Ryuichiro Ishitani, Mizuki Takemoto, and Kentaro Tomii.

## Install
This package run under the environment:
* Python: 3.10
* PyTorch: 2.0.1
* PyG: 2.5.3

The other package dependencies are described in [requirements.txt](/requirements.txt).

To install the package and dependencies:
```
pip install .
```

### Fpocket
Please install Fpocket version 4.2 in a location where the path is set.
In the following example, it is installed in /usr/local.
```
$ git clone https://github.com/Discngine/fpocket.git
$ cd fpocket
$ git checkout 4.2
$ make
$ sudo make install
```

## Usage
### Download dataset and model weights
Please download the model weights files from zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13070037.svg)](https://doi.org/10.5281/zenodo.13070037)
and save to the [examples](/examples) directory.

Extract the archive in the examples directory.
```
$ tar xJvf best_models.tar.xz
```
The five model files (fold0_best_model.pt~fold4_best_model.pt) will be extracted to the examples folder.


### Inference using PDB file
This repository includes a file named [1SQN.pdb](/examples/1SQN.pdb) as a sample.
If you will make predictions on this 1SQN.pdb file using the model downloaded from Zenodo as instructed above, there is no need to make any specific edits to the configuration file.

If you intend to make predictions on your own PDB file, prepare the PDB file for the target protein you want to predict.
First, remove water molecules, ligands, and other non-amino-acid residues.
Then, modify [run_pdb_infer.sh](/examples/run_pdb_infer.sh#L18) as described below.
```
sampler.pdb_files=["<Path to PDB file>"] \
```

To carry out predictions on multiple PDB files at once, make the following adjustments to the [run_pdb_infer.sh](/examples/run_pdb_infer.sh#L18) file.
```
sampler.pdb_files=["<Path to PDB file1>","<Path to PDB file2>"] \
```
Make sure there are no spaces before or after commas or brackets.
Run run_pdb_infer.sh as follows
```
bash run_pdb_infer.sh
```
After the computation is finished, a CSV file named infer_results.csv will be generated. This CSV file contains the following columns:
- PDB_ID: Name of the input file
- pred_0~pred_4: Predicted values (ranging from 0 to 1) for each model, where higher values indicate better predictions.
- pred_aver: Average of the predicted values for each model
- pred_std: Variance of the predicted values for each model


## Citation
If you find our work relevant to your research, please cite:
```
@article{ishitani2024pokeformer,
    title={Protein ligand binding site prediction using graph transformer neural network},
    author={Ryuichiro Ishitani, Mizuki Takemoto, Kentaro Tomii},
    year={2024},
    journal={}
}
```
