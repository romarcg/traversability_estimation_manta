# Traversability estimation

This repository includes simulated data from traversability robot traversability and code to train a CNN traversability estimator.

## Dataset

Contains data from simulated trajectories in v-rep simulator. Data is in cvs. A `meta_` file stores all the corresponding `cvs` files and `png` heightmaps for a each trajectory.

Heightmaps are store in `heightmaps` folder.

## Code

Contains scripts for generating traversability datasets training a CNN estimator and estimating traversability on evaluation heightmaps.

## Results

Here are some results from traversability on different heightmaps:


## Requirements

In order to test the source code, several frameworks/libraries are needed.
- python 3.5.3
- numpy 1.12.1
- matplotlib 2.0.0
- pandas 0.19.2
- tensorflow-gpu 1.0
- keras 2.0.3
- scikit-learn 0.18.1
- scikit-image 0.13.0
- joblib 0.11
