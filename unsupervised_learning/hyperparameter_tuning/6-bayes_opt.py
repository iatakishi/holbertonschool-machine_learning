#!/usr/bin/env python3
"""
Hyperparameter optimization using GPyOpt on a neural network model.
Optimizes 5 hyperparameters, saves tracking reports, implements early stopping,
checkpoints the best architecture, and plots optimization convergence.
"""

import os
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Load data locally to avoid internet reliance during testing
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale inputs for stable neural network training
X_train, X_val = X_train / 16.0, X_val / 16.0


# 2. Define the objective function for GPyOpt
def optimize_model(hyperparameters):
    """
    Trains a neural network given a 2D array of hyperparameters from GPyOpt.
    Returns the validation loss (satisficing metric) to be minimized.
    """
    # GPyOpt passes parameters as a 2D array matrix shape (1, D)
    params = hyperparameters[0]

    lr = float(params[0])
    num_units = int(params[1])
    dropout_rate = float(params[2])
    l2_reg = float(params[3])
    batch_size = int(params[4])

    # Build sequential neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            input_shape=(64,)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile with Adam optimizer using the sampled learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Establish descriptive filename template for hyperparameter checkpointing
    checkpoint_name = (
        f"model_lr{lr:.4f}_units{num_units}_drop{dropout_rate:.2f}"
        f"_l2{l2_reg:.5f}_batch{batch_size}.keras"
    )

    # Callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Get best validation loss achieved during training loop execution
    best_val_loss = min(history.history['val_loss'])

    # If checkpoint wasn't saved or initialized properly, clean up stray files
    # but preserve the actual best file by analyzing overall minimization later
    return best_val_loss


# 3. Setup the Hyperparameter Space Bounds
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.00001, 0.001)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
]

if __name__ == '__main__':
    # Force reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 4. Initialize and Run Bayesian Optimization
    print("Initializing Bayesian Optimization via GPyOpt...")
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=optimize_model,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        maximize=False
    )

    print("Running optimization iterations (Max: 30)...")
    optimizer.run_optimization(max_iter=30)

    # 5. Extract Optimal Hyperparameters
    best_x = optimizer.x_opt
    best_y = optimizer.fx_opt

    # 6. Save a concise report to 'bayes_opt.txt'
    print("Writing evaluation summary report to bayes_opt.txt...")
    with open('bayes_opt.txt', 'w') as f_report:
        f_report.write("Bayesian Optimization Report (GPyOpt)\n")
        f_report.write("=====================================\n\n")
        f_report.write(f"Optimized Val Loss (Satisficing Metric): {best_y[0]:.6f}\n\n")
        f_report.write("Optimal Hyperparameters Found:\n")
        f_report.write(f" - Learning Rate: {best_x[0]:.6f}\n")
        f_report.write(f" - Hidden Layer Units: {int(best_x[1])}\n")
        f_report.write(f" - Dropout Rate: {best_x[2]:.4f}\n")
        f_report.write(f" - L2 Regularization Weight: {best_x[3]:.6f}\n")
        f_report.write(f" - Batch Size: {int(best_x[4])}\n")

    print("Optimization complete. Displaying convergence graph...")

    # 7. Plot convergence and save the figure
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')
    plt.show()
