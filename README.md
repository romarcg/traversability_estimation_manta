# Traversability estimation

This repository includes simulated data from traversability robot traversability and code to train a CNN traversability estimator.

## Dataset

Contains data from simulated trajectories in v-rep simulator. Data is in cvs. A `meta_` file stores all the corresponding `cvs` files and `png` heightmaps for a each trajectory.

Heightmaps are store in `heightmaps` folder.

## Code

Contains scripts for `generate`-ing traversability datasets `train`-ning a CNN estimator and `estimate`-ing traversability on evaluation heightmaps.

## Results

Here are some results from traversability on different heightmaps:


## Requirements

In order to test the source code, several frameworks/libraries are needed:
- python > 3
- pandas
- tensorflow
- keras
- scikit-learn
- scikit-image
- joblib
