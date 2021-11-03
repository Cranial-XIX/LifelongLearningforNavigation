# A Lifelong Navigation Approach to Mobile Robot Navigation
This is the source code for [this paper](https://ieeexplore.ieee.org/document/9345478) **(RAL/ICRA 2021)**


Enter the following to see the arguments.
```
python main.py --help
```

## Make Data
In particular, type
```
python main.py --mode=make-data
```
to generate training data. Data files should be listed in ./data folder in the form task{i}-lidar.csv, task{i}-cmd.csv.


## Learning
Type
```
python main.py --mode=lifelong-learn
```
to learn the model. The learned model will be saved by default to ./models/trained_agent.pt


## Deployment
Type
```
python main.py --mode=predict-example
```
to see a particular example of loading the trained agent and predict the motion command from a lidar beam.

## Citation
If you find our paper interesing or the repo useful, please consider cite [this paper](https://ieeexplore.ieee.org/document/9345478)
```
@article{liu2021lifelong,
  title={A lifelong learning approach to mobile robot navigation},
  author={Liu, Bo and Xiao, Xuesu and Stone, Peter},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={1090--1096},
  year={2021},
  publisher={IEEE}
}
```
