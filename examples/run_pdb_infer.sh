#!/bin/bash
set -eux

GPU=0
TOPDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"/.. &> /dev/null && pwd )
MODELDIR=/path/to/model/location
RESULT=.

CONFIG_YAML=config_pdbinfer.yaml
PYTHON="python3 -u"

env PYTHONPATH=$TOPDIR \
    $PYTHON $TOPDIR/scripts/inference.py \
    yaml=$CONFIG_YAML \
    sampler.gpu=$GPU \
    sampler.model_path_list=[$MODELDIR/fold0/best_model.pt,$MODELDIR/fold1/best_model.pt,$MODELDIR/fold2/best_model.pt,$MODELDIR/fold3/best_model.pt,$MODELDIR/fold4/best_model.pt] \
    sampler.out_csv=$RESULT/infer_results.csv \
    sampler.pdb_files=["1SQN.pdb"] \
    sampler.pocket_result_base=$RESULT/pockets/
