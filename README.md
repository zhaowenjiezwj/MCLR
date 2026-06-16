**Environment Setup**

We recommend using Anaconda/Miniconda to manage the environment, ensuring correct C-level dependencies for meteorological data processing (e.g., `xarray`, `netCDF4`).

**Requirements**

`conda env create -f environment.yml`



**Quick Start**

The `train.py` script is structured using PyTorch Lightning. And the `era5_dataset.py` is used to preprocess the dataset.

To start a demonstration training run:

Example:

`python train.py \
    --data_dir ./dataset_sample \
    --batch_size 16 \
    --num_workers 4`