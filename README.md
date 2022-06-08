# Traffic

Traffic prediction on the PEMS-BAY dataset using a spatio-temporal graph convolutional variational autoencoder.

![Visualized predictions](https://github.com/aramuk/traffic/blob/main/figures/predictions.gif)

## Installation

`pip3 install -r requirements.txt`

## Results

| Experiment | MAE (val) | RMSE (val) | MAE (test) | RMSE (test) |
|---|---|---|---|---|
| `loss_x_all_decoder`  |  4.2003 |  6.4123 | 4.0664 | 5.9022 |
| `loss_x_start_decoder`  | 3.7063 | 6.1540 | 3.5463 | 5.5596 |