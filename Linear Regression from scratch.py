import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Class for splitting data into training and test sets
class DataSplitter:
    def __init__(self, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df):
        # Shuffle and split the DataFrame into train and test
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        test_count = int(len(df) * self.test_size)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

# Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, lr=0.001, precision=0.0001, max_iter=100):
        self.lr = lr
        self.precision = precision
        self.max_iter = max_iter
        self.W = None
        self.W_history = []
        self.cost_history = []
        self.X = None
        self.y = None

    def prepare_data(self, df):
        # Convert DataFrame to numpy arrays, add bias column
        cols = df.columns
        X = df[cols[:-1]]
        y = df[cols[-1]]
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        bias = np.ones((X.shape[0], 1))  # Add bias term
        X = np.hstack([bias, X])
        return X, y

    def cost_function(self, X, y, W):
        # Mean Squared Error
        return np.mean((X.dot(W) - y) ** 2)

    def compute_gradient(self, X, y, W):
        # Compute gradient for weights
        y_pred = X.dot(W)
        error = y_pred - y
        gradient = X.T.dot(error) / X.shape[0]
        return gradient

    def fit(self, df):
        # Train the model using gradient descent
        X, y = self.prepare_data(df)
        self.X = X
        self.y = y
        W = np.zeros(X.shape[1])
        self.W_history = [W.copy()]
        self.cost_history = [self.cost_function(X, y, W)]
        last_W = W + 100 * self.precision
        i = 1
        while i <= self.max_iter and np.linalg.norm(W - last_W) > self.precision:
            last_W = W.copy()
            grad = self.compute_gradient(X, y, W)
            W -= self.lr * grad
            self.W_history.append(W.copy())
            self.cost_history.append(self.cost_function(X, y, W))
            i += 1
        self.W = W
        return self

    def predict(self, X):
        # Predict target values for new data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        bias = np.ones((X.shape[0], 1))
        X = np.hstack([bias, X])
        return X.dot(self.W)

# Class for regression evaluation metrics
class EvaluationMetrics:
    @staticmethod
    def mse(y_true, y_pred):
        # Mean Squared Error
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        # Mean Absolute Error
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        # R-squared Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# Class for plotting results
class LinearRegressionVisualizer:
    def __init__(self, X, y, W_history, cost_history, final_weights):
        self.X = X
        self.y = y
        self.W_history = np.array(W_history)
        self.cost_history = cost_history
        self.final_weights = final_weights

    def plot_cost_function(self):
        # Plot cost (MSE) over iterations
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_history, color='#0072B2', marker='o', markersize=5, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_best_fit_line(self):
        # Plot regression line or predictions
        plt.figure(figsize=(8, 4))
        if self.X.shape[1] == 2:
            plt.scatter(self.X[:, 1], self.y, color='#56B4E9', label='Data', s=50, alpha=0.8)
            y_pred = self.X.dot(self.final_weights)
            sorted_idx = np.argsort(self.X[:, 1])
            plt.plot(self.X[sorted_idx, 1], y_pred[sorted_idx], color='#D55E00', linewidth=3, label='Best Fit')
            plt.xlabel('Feature', fontsize=12)
            plt.ylabel('Target', fontsize=12)
        else:
            y_pred = self.X.dot(self.final_weights)
            plt.scatter(self.y, y_pred, color='#56B4E9', label='Predictions', s=50, alpha=0.8)
            plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], color='#D55E00', linewidth=3, label='Perfect Fit')
            plt.xlabel('Actual', fontsize=12)
            plt.ylabel('Predicted', fontsize=12)
        plt.title('Best Fit Line / Predictions', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Linear Regression/Linear Regression/Student_Marks.csv')
    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df)

    # Train model
    model = LinearRegressionGD(lr=0.001, precision=0.0001, max_iter=100)
    model.fit(train_df)

    # Prepare train and test data for evaluation
    X_train, y_train = model.prepare_data(train_df)
    X_test, y_test = model.prepare_data(test_df)
    y_train_pred = model.predict(X_train[:, 1:] if X_train.shape[1] > 1 else X_train[:, 1])
    y_test_pred = model.predict(X_test[:, 1:] if X_test.shape[1] > 1 else X_test[:, 1])

    # Calculate and print metrics in a table
    train_mse = EvaluationMetrics.mse(y_train, y_train_pred)
    test_mse = EvaluationMetrics.mse(y_test, y_test_pred)
    train_mae = EvaluationMetrics.mae(y_train, y_train_pred)
    test_mae = EvaluationMetrics.mae(y_test, y_test_pred)
    train_r2 = EvaluationMetrics.r2(y_train, y_train_pred) * 100
    test_r2 = EvaluationMetrics.r2(y_test, y_test_pred) * 100

    metrics = [
        ["MSE", f"{train_mse:.4f}", f"{test_mse:.4f}"],
        ["MAE", f"{train_mae:.4f}", f"{test_mae:.4f}"],
        ["R² Score", f"{train_r2:.2f}%", f"{test_r2:.2f}%"]
    ]
    print("\nModel Evaluation Metrics:")
    print(tabulate(metrics, headers=["Metric", "Training Set", "Test Set"], tablefmt="fancy_grid"))

    print("\nFinal weights:", np.round(model.W, 4))

    # Show plots
    visualizer = LinearRegressionVisualizer(model.X, model.y, model.W_history, model.cost_history, model.W)
    visualizer.plot_cost_function()
    visualizer.plot_best_fit_line()