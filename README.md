# Archived project. No maintenance.

This project is not maintained anymore and is archived. Feel free to fork and make your own changes if needed.
It's because [Myo production and sales has officially ended as of Oct 12, 2018](https://support.getmyo.com/hc/en-us).

Thanks to everyone for their valuable feedback.


# myo-armband-nn
Gesture recognition using [myo armband](https://www.myo.com) via neural network (tensorflow library).
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
You can use your own scripts for collecting EMG data from Myo armband.
But you need to push 64-value array with data from each sensor.<br />
By default myo-python returns 8-value array from each sensors.
Each output return by 2-value array: ```[datetime, [EMG DATA]]```.<br />
64 - value array its 8 output from armband. Just put it to one dimension array.
So you just need to collect 8 values with gesture from armband (if you read data 10 times/s its not a problem).

In repo are collected dataset from Myo armband collected by me. Dataset contains only 5 gestures:
```
👍 - Ok    (1)
✊️ - Fist  (2)
✌️ - Like  (3)
🤘 - Rock  (4)
🖖 - Spock (5)
```

## Training network
```sh
python3 train.py
```
75k iteration take about 20 min on GTX 960 or 2h on i3-6100.

Accuracy after ~75k iteration (98.75%):
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-accuracy.png)

Loose after ~75k iteration (1.28):
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-losse.png)

## Prediction
### Prediction on data from MYO armband
```sh
python3 predict.py
```
You must have installed MYO SDK.
Script will return number (0-5) witch represent gesture (0 - relaxed arm).

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
I know that making prediction on training dataset wrong. But i don't have time to make testing dataset(


## Model
| **Fully connected 1 (528 neurons)** |
| :---: |
| ReLu |
| **Fully connected 2 (786 neurons)** |
| ReLu |
| **Fully connected 3 (1248 neurons)**  |
| ReLu |
| Dropout |
| **Softmax_linear** |
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-model.png)


## License
[GNU General Public License v3.0](https://github.com/exelban/myo-armband-nn/blob/master/LICENSE)
