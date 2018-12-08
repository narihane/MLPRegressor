# MLPRegressor

Neural Networks for Time Series Forcasting

Objective: 

  -Building a multilayer perceptron to forecast future natural gas prices

Dataset:

  -The original dataset is available via https://www.eia.gov/dnav/ng/hist/rngwhhdW.htm.

Tasks:

1. Use scikit MLPRegressor to train over the dataset.

2. Attempt different configurations for the regressor: 

   a) Attempt different numbers of hidden layers and different numbers of nodes in each hidden layer. Show 3 different
    configurations comparing their respective performance, and reasoning for the observed performance.

   b) Attempt 2 different learning rates, reasoning for the observed effect of varying the learning rate on the time before
   convergence.

Having attempted different configurations for the regressor, use the optimal configurations you reached for the following experiments:

3. Observe and report on the progress of both training and validation errors until the optimal performance is reached.

4. Take and report on adequate measures to guarantee that the regressor is not overfitted to the training data.


Bonus: 

  -Using the original file “ngp.csv” to build a time series of 4 inputs to forecast the price of one week into the future.
