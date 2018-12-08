import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Hyper parameters
test_size = 0.1
validation_size = 0.1
layers = (32, 8)
Lambda = 1e-05
batch_size = 64
initial_learning_rate = 0.01
epochs = 500


# Reading the timesereis_8_2 dataset
#df = pd.read_csv('../Datasets/timesereis_8_2.csv')
#
## Getting the input/output
#X = df[['0', '1', '2', '3', '4', '5', '6', '7']]
#Y = df[['8', '9']]

# Reading the timesereis_4_1 dataset
df = pd.read_csv('../Datasets/ngp.csv')['price'][1:-1][::-1]
df = df.dropna()
df = np.array([df[i:i + 5] for i in range(len(df) - 4)])

# Getting the input/output
X = df[:, :-1]
Y = df[:, -1:]

# Normalizing the data
X = preprocessing.scale(X)
Y = preprocessing.scale(Y)

# Split data into Train/Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1)

# Initializing the regressor model
regressor = MLPRegressor(
        hidden_layer_sizes=layers,
        activation='logistic',
        solver='adam',
        alpha=Lambda,
        batch_size=batch_size,
        learning_rate='adaptive',
        learning_rate_init=initial_learning_rate,
        max_iter=epochs,
        shuffle=True,
        tol=0.0001,
        verbose=False,
        early_stopping=True,
        validation_fraction=validation_size)

# Train the regressor
avg_score = 0
avg_error = 0
min_error = 1
max_error = 0
avg_conv_iter = 0
for i in range(10) :
    regressor.fit(X_train, Y_train) 
    avg_score += regressor.best_validation_score_
    avg_conv_iter += regressor.n_iter_
    cur_error = 1 - regressor.best_validation_score_
    avg_error += cur_error
    min_error = min(cur_error, min_error)
    max_error = max(cur_error, max_error)

avg_score /= 10
avg_error /= 10
avg_conv_iter /= 10

print('avg_score    :', round(avg_score, 2))
print('avg_error    :', round(avg_error, 2))
print('max_error    :', round(max_error, 2))
print('min_error    :', round(min_error, 2))
print('avg_conv_iter:', round(avg_conv_iter, 2))

print('Regressor Score:', regressor.score(X_test, Y_test))

# Plotting the gradient error curves
plt.plot(np.arange(0, len(regressor.loss_curve_)), regressor.loss_curve_, color='red', label='Training')
plt.plot(np.arange(0, len(regressor.validation_scores_)), 1 - np.array(regressor.validation_scores_), color='blue', label='Validation')
#
# setting title and labels
plt.title('Number of iterations VS Number of errors')
plt.xlabel('Number of iterations')
plt.ylabel('Number of errors')
plt.legend()

#
## displaying the plot
plt.show()
