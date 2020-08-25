# DNN implementation (without using libraries such as TensorFlow)
# Training and Prediction
# Author: eCabral87 (gcabral@u2py.mx)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix


class DNN:
    def __init__(self, config, data_set_x, data_set_y, validation_data=None, batch_size=None, accuracy=None):
        self.layers = len(config['layers'])
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.accuracy = accuracy
        self.nn_config = config
        self.X = data_set_x  # (nx, m)
        self.Y = data_set_y  # (ny, m)
        self.W, self.B = self.w_b_initialization()


    def w_b_initialization(self, kind='random'):
        if kind == 'random':
            # random initialization
            W, B = [], []
            for k in range(self.layers):
                if k > 0:
                    W.append(np.random.randn(self.nn_config['layers'][k], self.nn_config['layers'][k-1]) * 0.01)
                else:
                    W.append(np.random.randn(self.nn_config['layers'][k], self.X.shape[0])*0.01)
                B.append(np.zeros((self.nn_config['layers'][k], 1)))
            return W, B
        else:
            raise Exception('This kind of initialization is not implemented yet')
            # TODO: add more

    def plot_loss(self, loss, train_loss, test_loss=None):
        # plot loss during training
        plt.style.use('ggplot')
        plt.title(loss)
        plt.plot(train_loss, label='train')
        if test_loss is not None:
            plt.plot(test_loss, label='test')
        plt.legend()
        plt.show()

    def plot_regresion(self, y, y_pred=None):
        # plot regression points
        plt.style.use('ggplot')
        plt.title('Regresion')
        plt.plot(y, label='actual')
        if y_pred is not None:
            plt.plot(y_pred, label='prediction')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def function(self, fun, x):
        if fun == 'logistic':
            return 1 / (1 + np.exp(-x))
        elif fun == 'tanh':
            return np.tanh(x)
        elif fun == 'ReLu':
            return np.where(x >= 0, x, 0)
        elif fun == 'Leaky ReLu':
            return np.where(x >= 0, x, x * 0.01)
        elif fun == 'linear':
            return x
        elif fun == 'softmax':
            exps = np.exp(x - np.max(x))
            return exps / np.sum(exps, axis=0, keepdims=True)
        else:
            raise Exception('This activation function is not implemented yet')
            # TODO: add more

    def derivate_fun(self, fun, x):
        if fun == 'logistic':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif fun == 'tanh':
            return (1 - np.tanh(x))**2
        elif fun == 'ReLu':
            return np.where(x >= 0, 1, 0)
        elif fun == 'Leaky ReLu':
            return np.where(x >= 0, 1, 0.01)
        elif fun == 'linear':
            return np.ones(x.shape)
        elif fun == 'softmax':
            # TODO: test before apply it
            s = x.reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)
        else:
            raise Exception('This activation function is not implemented yet')
            # TODO: add more

    def forward_prop(self, X):
        Z, A = [], []
        for k in range(self.layers):  # A_k, Z_k : 1,2,..layers
            if k == 0:
                Z.append(np.dot(self.W[k], X) + self.B[k])
            else:
                Z.append(np.dot(self.W[k], A[k - 1]) + self.B[k])
            A.append(self.function(self.nn_config['activation_fun'][k], Z[k]))
        return Z, A

    def compute_loss_delta(self, A, Y, m):
        if self.nn_config['loss'] == 'mse':
            L = (0.5 / m) * np.sum(np.power(A[-1] - Y, 2))
            dA = A[-1] - Y
        elif self.nn_config['loss'] == 'binary_cross_entropy':
            L = -(Y * np.log(A[-1] + 1e-9) + (1 - Y) * np.log(1 - A[-1] + 1e-9))
            dA = -Y / (A[-1] + 1e-9) + (1 - Y) / (1 - A[-1] + 1e-9)
        elif self.nn_config['loss'] == 'categorical_cross_entropy':
            L = -np.sum(Y * np.log(A[-1] + 1e-9))/m
            dA = -Y / (A[-1] + 1e-9)
        else:
            raise Exception('This loss is not implemented')
        return L, dA

    def back_prop(self, dA, Z, A, X, Y, dw, db, m):
        for k in range(self.layers - 1, -1, -1):
            if self.nn_config['activation_fun'][k] == 'softmax':
                dZ = A[-1] - Y
            else:
                dZ = dA * self.derivate_fun(self.nn_config['activation_fun'][k], Z[k])

            if k == 0:
                dW = (1 / m) * np.dot(dZ, X.T)
            else:
                dW = (1 / m) * np.dot(dZ, A[k - 1].T)
            dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.W[k].T, dZ)

            # Get average of the batches
            dw[k] += dW
            db[k] += dB

        return dw, db

    def update_w_b_weights(self, dW, dB, VdW, VdB):
        for k in range(self.layers):
            VdW[k] = self.nn_config['momentum'] * VdW[k] + (1 - self.nn_config['momentum']) * dW[k]
            VdB[k] = self.nn_config['momentum'] * VdB[k] + (1 - self.nn_config['momentum']) * dB[k]
            self.W[k] = self.W[k] - self.nn_config['lr'] * VdW[k]
            self.B[k] = self.B[k] - self.nn_config['lr'] * VdB[k]

    def train(self):
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []
        history_w_b = []

        #  Obtain number of batches
        if isinstance(self.batch_size, int):
            if self.X.shape[1] % self.batch_size == 0:
                n_batches = int(self.X.shape[1] / self.batch_size)
            else:
                n_batches = int(self.X.shape[1] / self.batch_size) - 1
        else:
            n_batches = 1
            self.batch_size = self.X.shape[1]

        # Momentum variables
        vdw = [np.zeros(w.shape) for w in self.W]
        vdb = [np.zeros(b.shape) for b in self.B]

        for epoch in range(self.nn_config['epochs']):
            # Shuffle training set
            sort_examples = np.arange(self.X.shape[1])
            np.random.shuffle(sort_examples)
            x_train, y_train = self.X[:, sort_examples], self.Y[:, sort_examples]
            x_train = [x_train[:, self.batch_size * i:self.batch_size * (i + 1)] for i in range(0, n_batches)]
            y_train = [y_train[:, self.batch_size * i:self.batch_size * (i + 1)] for i in range(0, n_batches)]
            # Initialize delta weights
            dw_batch = [np.zeros(w.shape) for w in self.W]
            db_batch = [np.zeros(b.shape) for b in self.B]

            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []

            for batch_x, batch_y in zip(x_train, y_train):
                # Forward propagation
                Z, A = self.forward_prop(batch_x)

                # Calculate loss
                L, dA = self.compute_loss_delta(A, batch_y, self.batch_size)

                # Backward propagation
                dw_batch, db_batch = self.back_prop(dA, Z, A, batch_x, batch_y, dw_batch, db_batch, self.batch_size)

                # Save metrics
                train_losses.append(L)
                try:
                    train_accuracies.append(accuracy_score(batch_y.T, np.where(A[-1] > 0.5, 1, 0).T))
                except ValueError:
                    pass

                if self.validation_data is not None:
                    a_test_pred = self.forward_prop(self.validation_data[0])[1]
                    y_test = self.validation_data[1]
                    test_losses.append(self.compute_loss_delta(a_test_pred, y_test, y_test.shape[1])[0])
                    try:
                        test_accuracies.append(accuracy_score(y_test.T, np.where(a_test_pred[-1]> 0.5, 1, 0).T))
                    except ValueError:
                        pass

            #  Compute updates
            self.update_w_b_weights(dw_batch, db_batch, vdw, vdb)

            # History metrics
            history_train_losses.append(np.mean(train_losses))
            history_test_losses.append(np.mean(test_losses))
            history_w_b.append((self.W, self.B))

            if self.accuracy is not None:
                history_train_accuracies.append(np.mean(train_accuracies))
                history_test_accuracies.append(np.mean(test_accuracies))

            # Verbose
            if self.nn_config['verbose']:
                if self.validation_data is not None:
                    if self.accuracy is None:
                        print('Epoch {} / {} | train loss: {} | val loss : {} | '.format(
                                epoch, self.nn_config['epochs'], np.round(np.mean(train_losses), 3),
                                np.round(np.mean(test_losses), 3)))
                    else:
                        print('Epoch {} / {} | train loss: {} | train accuracy: {} | val loss : {} | val accuracy : {} '.format(
                            epoch, self.nn_config['epochs'], np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3),
                            np.round(np.mean(test_losses), 3), np.round(np.mean(test_accuracies), 3)))
                else:
                    if self.accuracy is None:
                        print('Epoch {} / {} | train loss: {} '.format(epoch, epochs, np.round(np.mean(train_losses), 3)))
                    else:
                        print('Epoch {} / {} | train loss: {} | train accuracy: {} '.format(
                                epoch, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3)))

        history = {'epochs': self.nn_config['epochs'],
                   'train_loss': history_train_losses,
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies,
                   'weigths_w_c': history_w_b
                   }
        return history


if __name__ == "__main__":
    np.random.seed(1234)  # set global seed to random calls
    # Regression example
    data_x, data_y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
    # standardize dataset
    data_x = StandardScaler().fit_transform(data_x)
    data_y = StandardScaler().fit_transform(data_y.reshape(len(data_y), 1))
    data_x, data_y = data_x.T, data_y.T

    # Binary classification
    # data_x, data_y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # data_x, data_y = data_x.T, data_y.reshape(1, len(data_y))
    # # select indices of points with each class label
    # for i in range(2):
    #     samples_ix = np.where(data_y == i)
    #     plt.scatter(data_x[0, samples_ix], data_x[1, samples_ix], label=str(i))
    # plt.legend()
    # plt.show()

    # Multi-class classification
    # data_x, data_y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    # # select indices of points with each class label
    # for i in range(3):
    #     samples_ix = np.where(data_y == i)
    #     plt.scatter(data_x[samples_ix, 0], data_x[samples_ix, 1])
    # plt.show()
    # from tensorflow.keras.utils import to_categorical
    # # one hot encode output variable
    # tmp = data_y
    # data_y = to_categorical(data_y, dtype='int32')
    # data_x, data_y = data_x.T, data_y.T


    # split into train and test
    n_train = 500
    train_x, test_x = data_x[:, :n_train], data_x[:, n_train:]
    train_y, test_y = data_y[:, :n_train], data_y[:, n_train:]

    nn_config = {'layers': [25, 1], 'activation_fun': ['ReLu', 'linear'],
                 'initialization': ['random', 'random'], 'loss': 'mse',  # 'mse, categorical_cross_entropy, binary_cross_entropy'
                 'epochs': 100, 'verbose': True,
                 'lr': 0.01, 'momentum': 0.9}
    dnn = DNN(nn_config, train_x, train_y, validation_data=[test_x, test_y], accuracy=None, batch_size=32)
    logs = dnn.train()
    dnn.plot_loss(nn_config['loss'], logs['train_loss'], logs['test_loss'])
    y_true, y_pred = test_y[0], dnn.forward_prop(test_x)[1][-1][0]
    dnn.plot_regresion(y_true, y_pred)
    print('mse on test set: ', mean_squared_error(y_true, y_pred))
    
    # dnn.plot_loss('accuracy', logs['train_acc'], logs['test_acc'])
    # cm = confusion_matrix(test_y.T.argmax(axis=1), np.where(dnn.forward_prop(test_x)[1][-1] > 0.5, 1, 0).T.argmax(axis=1))  # FIX (check) for multi-class
    # cm = confusion_matrix(test_y.T, np.where(dnn.forward_prop(test_x)[1][-1] > 0.5, 1, 0).T)  # binary class
    # print('Confusion matrix:\n', cm)
    # dnn.plot_confusion_matrix(cm, ['class1', 'class2'])
