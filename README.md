# Linear Regression (From Scratch)

Welcome to the **Linear Regression** project! This folder is a hands-on, beginner-friendly mini-lab for learning and experimenting with linear regression using Python and NumPy — no machine learning libraries required.

## 📂 Folder Contents

- **Linear Regression from scratch copy.py**
  - A complete Python script that walks you through the entire linear regression workflow:
    - Data loading and preparation
    - Train/test splitting
    - Model training using gradient descent
    - Evaluation with common metrics (MSE, MAE, R²)
    - Visualization of cost function and regression results
    - Clean, modular code with comments and classes

- **Student_Marks.csv**
  - A real-world style dataset with two columns: features (e.g., study hours) and target (student marks). This is used to train and test the model.

---

## 🚀 How It Works

1. **Data Loading**: Reads the CSV file into a pandas DataFrame.
2. **Splitting**: Uses a custom class to split the data into training and test sets (default: 80% train, 20% test).
3. **Training**: Fits a linear regression model from scratch using gradient descent (no sklearn!).
4. **Evaluation**: Calculates MSE, MAE, and R² for both training and test sets, and prints them in a clear table.
5. **Visualization**: Plots the cost function over iterations and the regression line (or predictions) vs. actual data.

---

## 🛠️ How to Run

1. Make sure you have Python 3, numpy, pandas, matplotlib, and tabulate installed.
2. Open a terminal in this folder and run:
   ```bash
   python "Linear Regression from scratch copy.py"
   ```
3. Check the console for a summary table of metrics and the final weights.
4. Enjoy the automatic plots for cost and regression fit!

---

## ✏️ Customization & Experimentation

- **Try your own data**: Replace `Student_Marks.csv` with your own CSV (same format: features then target).
- **Change model settings**: Tweak learning rate, number of iterations, or test size in the script.
- **Explore the code**: The script is modular and well-commented. Try changing the cost function, adding features, or visualizing more!

---

## 📚 What You'll Learn
- How linear regression works under the hood
- How to implement gradient descent
- How to split data and evaluate a model
- How to visualize results and interpret metrics

---

**Have fun learning and experimenting!** 