# lifelong_robotics

Enter the following to see the arguments.
```
python main.py --help
```
# make data
In particular, type
```
python main.py --mode=make-data
```
to generate training data. Data files should be listed in ./data folder in the form task{i}-lidar.csv, task{i}-cmd.csv.

# learning
Type
```
python main.py --mode=lifelong-learn
```
to learn the model. The learned model will be saved by default to ./models/trained_agent.pt

# deployment example
Type
```
python main.py --mode=predict-example
```
to see a particular example of loading the trained agent and predict the motion command from a lidar beam.
