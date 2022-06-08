#!/bin/bash

script_dir=$(dirname "$0")
proj_dir=$(realpath "$script_dir/")

cd "$proj_dir"
echo "Zipping directory: $(pwd)"

zip -r 'group17.zip' . \
    -x  "datasets/METR-LA/*" \
        "datasets/PEMS-BAY/train.h5" \
        "datasets/PEMS-BAY/val.h5" \
        "datasets/PEMS-BAY/test.h5" \
        ".git/*" \
        "**/__pycache__/*"
