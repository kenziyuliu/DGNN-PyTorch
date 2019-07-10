#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Parameters: 'vera' or 'mm'"
    exit
fi

# Remove useless pycaches
rm -rf graph/__pycache__/
rm -rf model/__pycache__/

if [ $1 == "vera" ]; then
    scp -P 51000 -r model config feeders graph main.py ensemble.py root@10.66.31.100:/workplace/ken/repos/agcn-all/2s-AGCN
elif [ $1 == 'mm' ]; then
    scp -r model config feeders graph main.py ensemble.py zliu6676@129.78.10.182:/home2/zliu6676/action-recognition/agcn-all/2s-AGCN
else
    echo 'Target does not match one of the supported targets'
fi
