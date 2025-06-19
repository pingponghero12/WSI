## CUDA
time python3 mnistcuda.py 
MNIST Neural Network Training with CUDA
CUDA module imported
Loading MNIST dataset from sklearn...
Training data shape: (60000, 784)
Test data shape: (10000, 784)
Training labels range: 0 to 9
Starting training...
Training data shape: (784, 2000)
Initializing CUDA Neural Network...
Input size: 784
Hidden size: 128
Output size: 10
Batch size: 2000
CUDA Neural Network initialized successfully!
Starting training for 2000 epochs with learning rate 0.35
Epoch 0, Accuracy: 0.278
Epoch 50, Accuracy: 0.917
Epoch 100, Accuracy: 0.9455
Epoch 150, Accuracy: 0.963
Epoch 200, Accuracy: 0.975
Epoch 250, Accuracy: 0.985
Epoch 300, Accuracy: 0.9915
Epoch 350, Accuracy: 0.9955
Epoch 400, Accuracy: 0.998
Epoch 450, Accuracy: 0.999
Epoch 500, Accuracy: 0.9995
Epoch 550, Accuracy: 0.9995
Epoch 600, Accuracy: 0.9995
Epoch 650, Accuracy: 1
Epoch 700, Accuracy: 1
Epoch 750, Accuracy: 1
Epoch 800, Accuracy: 1
Epoch 850, Accuracy: 1
Epoch 900, Accuracy: 1
Epoch 950, Accuracy: 1
Epoch 1000, Accuracy: 1
Epoch 1050, Accuracy: 1
Epoch 1100, Accuracy: 1
Epoch 1150, Accuracy: 1
Epoch 1200, Accuracy: 1
Epoch 1250, Accuracy: 1
Epoch 1300, Accuracy: 1
Epoch 1350, Accuracy: 1
Epoch 1400, Accuracy: 1
Epoch 1450, Accuracy: 1
Epoch 1500, Accuracy: 1
Epoch 1550, Accuracy: 1
Epoch 1600, Accuracy: 1
Epoch 1650, Accuracy: 1
Epoch 1700, Accuracy: 1
Epoch 1750, Accuracy: 1
Epoch 1800, Accuracy: 1
Epoch 1850, Accuracy: 1
Epoch 1900, Accuracy: 1
Epoch 1950, Accuracy: 1
Training completed!
Public forward returning 20000 predictions

Final training accuracy: 1.0000

Preparing test data...
Running inference on test data...
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Public forward returning 20000 predictions
Test Accuracy: 0.9004

real	2m7.064s
user	2m5.159s
sys	0m1.423s

## Python
time python3 mnistpy.py 
Loading MNIST data...
Loading MNIST dataset from sklearn...
Training data shape: (60000, 784)
Test data shape: (10000, 784)
Training labels range: 0 to 9
Starting training...
Training data shape: (784, 2000)
Iteration:  0
Accuracy:  0.109
Iteration:  50
Accuracy:  0.6535
Iteration:  100
Accuracy:  0.7665
Iteration:  150
Accuracy:  0.815
Iteration:  200
Accuracy:  0.839
Iteration:  250
Accuracy:  0.8635
Iteration:  300
Accuracy:  0.887
Iteration:  350
Accuracy:  0.8995
Iteration:  400
Accuracy:  0.9205
Iteration:  450
Accuracy:  0.932
Iteration:  500
Accuracy:  0.9405
Iteration:  550
Accuracy:  0.949
Iteration:  600
Accuracy:  0.956
Iteration:  650
Accuracy:  0.9625
Iteration:  700
Accuracy:  0.968
Iteration:  750
Accuracy:  0.973
Iteration:  800
Accuracy:  0.9765
Iteration:  850
Accuracy:  0.981
Iteration:  900
Accuracy:  0.9815
Iteration:  950
Accuracy:  0.9845
Iteration:  1000
Accuracy:  0.986
Iteration:  1050
Accuracy:  0.989
Iteration:  1100
Accuracy:  0.9895
Iteration:  1150
Accuracy:  0.991
Iteration:  1200
Accuracy:  0.992
Iteration:  1250
Accuracy:  0.993
Iteration:  1300
Accuracy:  0.9955
Iteration:  1350
Accuracy:  0.996
Iteration:  1400
Accuracy:  0.996
Iteration:  1450
Accuracy:  0.9965
Iteration:  1500
Accuracy:  0.998
Iteration:  1550
Accuracy:  0.9985
Iteration:  1600
Accuracy:  0.999
Iteration:  1650
Accuracy:  0.999
Iteration:  1700
Accuracy:  0.999
Iteration:  1750
Accuracy:  0.999
Iteration:  1800
Accuracy:  0.999
Iteration:  1850
Accuracy:  0.9995
Iteration:  1900
Accuracy:  1.0
Iteration:  1950
Accuracy:  1.0

Preparing test data...
Running inference on test data...
Test Accuracy:  0.7791

real	1m52.905s
user	6m11.680s
sys	0m1.969s

## Python with Xavier init

time python3 mnistpy.py 
Loading MNIST data...
Loading MNIST dataset from sklearn...
Training data shape: (60000, 784)
Test data shape: (10000, 784)
Training labels range: 0 to 9
Starting training...
Training data shape: (784, 2000)
Iteration:  0
Accuracy:  0.1185
Iteration:  50
Accuracy:  0.7275
Iteration:  100
Accuracy:  0.8195
Iteration:  150
Accuracy:  0.8585
Iteration:  200
Accuracy:  0.878
Iteration:  250
Accuracy:  0.889
Iteration:  300
Accuracy:  0.8975
Iteration:  350
Accuracy:  0.9035
Iteration:  400
Accuracy:  0.9105
Iteration:  450
Accuracy:  0.9145
Iteration:  500
Accuracy:  0.921
Iteration:  550
Accuracy:  0.922
Iteration:  600
Accuracy:  0.927
Iteration:  650
Accuracy:  0.9305
Iteration:  700
Accuracy:  0.9345
Iteration:  750
Accuracy:  0.9375
Iteration:  800
Accuracy:  0.94
Iteration:  850
Accuracy:  0.9425
Iteration:  900
Accuracy:  0.9435
Iteration:  950
Accuracy:  0.945
Iteration:  1000
Accuracy:  0.947
Iteration:  1050
Accuracy:  0.949
Iteration:  1100
Accuracy:  0.9515
Iteration:  1150
Accuracy:  0.9545
Iteration:  1200
Accuracy:  0.956
Iteration:  1250
Accuracy:  0.9605
Iteration:  1300
Accuracy:  0.963
Iteration:  1350
Accuracy:  0.964
Iteration:  1400
Accuracy:  0.965
Iteration:  1450
Accuracy:  0.967
Iteration:  1500
Accuracy:  0.968
Iteration:  1550
Accuracy:  0.968
Iteration:  1600
Accuracy:  0.968
Iteration:  1650
Accuracy:  0.969
Iteration:  1700
Accuracy:  0.9705
Iteration:  1750
Accuracy:  0.9715
Iteration:  1800
Accuracy:  0.9735
Iteration:  1850
Accuracy:  0.974
Iteration:  1900
Accuracy:  0.9745
Iteration:  1950
Accuracy:  0.975

Preparing test data...
Running inference on test data...
Test Accuracy:  0.8941

real	1m32.479s
user	5m34.056s
sys	0m1.134s
