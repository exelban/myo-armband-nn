# myo-armband-nn
Gesture recognition using myo armband via neural network (tensorflow library).
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-logo.jpg)


## Requirement
**Library** | **Version**
--- | ---
**Python** | **^3.5**
**Tensorflow** | **^1.1.0** 
**Numpy** | **^1.12.0**
**sklearn** |  **^0.18.1**
**[myo-python](https://github.com/NiklasRosenstein/myo-python)** |  **^0.2.2**


## Collecting data


## Training network
### Train network
```sh
python3 train.py
```
75k iteration take about 20 min on GTX 960 or 2h on i3-6100. (17:24)



## Prediction
### Prediction on data from MYO armband
```sh
python3 predict.py
```
You must have installed MYO SDK.

### Prediction on training dataset
```sh
python3 predict_train_dataset.py
```
Example output:
```
Accuracy on Test-Set: 98.27% (19235 / 19573)
[2438    5    9    6    4   20] (0) Relax
[   4 2652   45    1    3    9] (1) Ok
[   8   44 4989    1    1    9] (2) Fist
[   8    2    2 4152   28   13] (3) Like
[   2    5    6   27 1839    1] (4) Rock
[  14   22   13   21    5 3165] (5) Spock
 (0) (1) (2) (3) (4) (5)
```


## License
[GNU General Public License v3.0](https://github.com/exelban/myo-armband-nn/blob/master/LICENSE)
