import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import os
import time
from scipy.stats import binned_statistic_2d
import optuna

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def reweight(events, model, batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    epsilon = 1e-50  # A small value to prevent division by zero
    f = np.clip(f, epsilon, 1 - epsilon)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

# Weighted binary crossentropy for classifying two samples with weights
def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1)  # event weights
    y_true = tf.gather(y_true, [0], axis=1)   # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))

    return K.mean(t_loss)

def omnifold(theta0_S, initial_weights, smeared_weights, theta_unknown_S, iterations, model_builder, verbose=0):
    # Since we lack theta0_G (generator-level data), we'll assume theta0_S serves as both theta0_G and theta0_S
    # This is a limitation and may affect the unfolding accuracy

    weights = np.empty(shape=(iterations, 2, len(theta0_S)))
    # shape = (iteration, step, event)

    # Use theta0_S as both generator-level and detector-level data
    theta0_G = theta0_S.copy()
    theta0_S = theta0_S.copy()

    # Labels for the classifier
    labels0 = np.zeros(len(theta0_S))
    labels1 = np.ones(len(theta0_S))
    labels_unknown = np.ones(len(theta_unknown_S))

    # Prepare data for Step 1
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    # Initialize weights with smeared_weights for detector-level and initial_weights for generator-level
    weights_pull = smeared_weights.copy()
    weights_push = smeared_weights.copy()

    for i in range(iterations):

        if verbose > 0:
            print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: Reweight detector-level simulated data to match observed data
        if verbose > 0:
            print("STEP 1\n")

        # Weights for training: weights_push for simulated data, ones for observed data
        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        # Split the data for training and validation
        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1, test_size=0.2)

        # Stack labels and weights
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

        # Build and compile a new model for Step 1
        model_step1 = model_builder()
        model_step1.compile(loss=weighted_binary_crossentropy,
                            optimizer='Adam',
                            metrics=['accuracy'])

        #
        earlystopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )

        # Train the model
        model_step1.fit(
            X_train_1,
            Y_train_1,
            sample_weight=w_train_1,
            epochs=200,
            batch_size=10000,
            validation_data=(X_test_1, Y_test_1, w_test_1),
            callbacks=[earlystopping],
            verbose=verbose
        )

        # Update weights_pull
        weights_pull = weights_push * reweight(theta0_S, model_step1)
        weights[i, 0, :] = weights_pull

        # Clear model to free memory
        del model_step1
        tf.keras.backend.clear_session()
        gc.collect()

        # STEP 2: Reweight generator-level data based on weights_pull
        if verbose > 0:
            print("\nSTEP 2\n")

        # Prepare labels for Step 2
        labels0 = np.zeros(len(theta0_G))
        labels1 = np.ones(len(theta0_G))

        # Prepare data for Step 2
        xvals_2 = np.concatenate((theta0_G, theta0_G))
        yvals_2 = np.concatenate((labels0, labels1))
        weights_2 = np.concatenate((initial_weights, weights_pull))

        # Split the data for training and validation
        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(
            xvals_2, yvals_2, weights_2, test_size=0.2)

        # Stack labels and weights
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)

        # Build and compile a new model for Step 2
        model_step2 = model_builder()
        model_step2.compile(loss=weighted_binary_crossentropy,
                            optimizer='Adam',
                            metrics=['accuracy'])

        # Define EarlyStopping callback
        earlystopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )

        # Train the model
        model_step2.fit(
            X_train_2,
            Y_train_2,
            sample_weight=w_train_2,
            epochs=200,
            batch_size=10000,
            validation_data=(X_test_2, Y_test_2, w_test_2),
            callbacks=[earlystopping],
            verbose=verbose
        )

        # Update weights_push
        weights_push = weights_pull * reweight(theta0_G, model_step2)
        weights[i, 1, :] = weights_push

        # Clear model to free memory
        del model_step2
        tf.keras.backend.clear_session()
        gc.collect()

        # Optional delay
        time.sleep(1)

    return weights

# Load your datasets
theta0_S = np.load('theta0_S.npy')                # Detector-level synthetic data
initial_weights = np.load('initial_weights.npy')  # Weights for truth-level data
smeared_weights = np.load('smeared_weights.npy')  # Weights for detector-level data
theta_unknown_S = np.load('theta_unknown_S.npy')  # Observed experimental data

# Import comparison distribution
comparing_weights = np.load('initial_weights_real.npy')

# Create the output directory if it doesn't exist
output_dir = 'iterative_output'
os.makedirs(output_dir, exist_ok=True)

# Define edges for the heatmap bins based on the data range
x_edges = np.linspace(np.min(theta0_S[:, 0]), np.max(theta0_S[:, 0]), 50)
y_edges = np.linspace(np.min(theta0_S[:, 1]), np.max(theta0_S[:, 1]), 50)

# Define the create_heatmap function outside the loop
def create_heatmap(data, weights, title, ax, color_map='RdYlGn_r'):
    statistic, _, _, _ = binned_statistic_2d(
        data[:, 0], data[:, 1], weights, statistic='mean', bins=[x_edges, y_edges]
    )
    statistic = np.log10(statistic)
    heatmap = ax.imshow(
        statistic.T, 
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
        origin='lower', 
        cmap=color_map, 
        aspect='equal'
    )
    scatter = ax.scatter(
        data[:, 0], data[:, 1], 
        c=np.log10(weights), cmap=color_map, 
        s=4, label='Data Points'
    )
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    return heatmap

# Initialize variables to keep track of the best results
best_difference = np.inf
best_params = None
best_weights = None

def objective(trial):
    global best_difference, best_weights, best_params

    print(f"Running trial {trial.number + 1}")

    # Suggest parameters using Bayesian optimization
    num_layers = trial.suggest_int('num_layers', 1, 5)
    omnifold_iterations = trial.suggest_int('omnifold_iterations', 1, 5)

    # Suggest nodes per layer for up to 5 layers
    nodes_per_layer = []
    for i in range(5):
        if i < num_layers:
            nodes = trial.suggest_int(f'layer_{i}_nodes', 32, 256)
        else:
            nodes = 0
        nodes_per_layer.append(nodes)

    # Define model architecture
    def model_builder():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(theta0_S.shape[1],)))
        for nodes in nodes_per_layer:
            if nodes > 0:
                model.add(tf.keras.layers.Dense(
                    nodes,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4)
                ))
                model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    # Run the Omnifold algorithm
    weights_full = omnifold(
        theta0_S,
        initial_weights,
        smeared_weights,
        theta_unknown_S,
        omnifold_iterations,
        model_builder,
        verbose=0  # Set verbose to 0 to reduce output during iterations
    )

    # Extract the weights from the last iteration
    weights_last = weights_full[-1, -1, :]

    # Normalize weights and compute the difference
    weights_min, weights_max = np.min(weights_last), np.max(weights_last)
    init_weights_min, init_weights_max = np.min(comparing_weights), np.max(comparing_weights)
    normalised_weights = init_weights_min + (weights_last - weights_min) / (weights_max - weights_min) * (init_weights_max - init_weights_min)
    difference_distribution = np.abs(normalised_weights - comparing_weights)
    difference = np.sum(difference_distribution)

    print(f"Difference: {difference}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    heatmaps = []

    # Plot Initial Weights
    heatmaps.append(create_heatmap(
        theta0_S, comparing_weights, 'Initial Weights', axes[0], 'magma'
    ))

    # Plot Normalized Unfolded Weights
    heatmaps.append(create_heatmap(
        theta0_S, normalised_weights, 'Normalized Unfolded Weights', axes[1], 'magma'
    ))

    # Plot Difference Distribution
    heatmaps.append(create_heatmap(
        theta0_S, difference_distribution, 'Normalized Unfolded Weights - Initial Weights', axes[2], 'RdYlGn_r'
    ))

    # Add colorbars
    for ax, hm in zip(axes.flat, heatmaps):
        cbar = fig.colorbar(hm, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Log10(Statistic)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    nodes_str = '_'.join(str(n) for n in nodes_per_layer if n > 0)
    plot_filename = os.path.join(output_dir, f'trial_{trial.number + 1}_layers_{num_layers}_nodes_{nodes_str}_omni_{omnifold_iterations}_score_{difference:.3g}.png')
    data_filename = os.path.join(output_dir, f'trial_{trial.number + 1}_layers_{num_layers}_nodes_{nodes_str}_omni_{omnifold_iterations}_score_{difference:.3g}.npy')
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free memory
    np.save(data_filename, weights_full)
    
    """
    if 0.156 <= difference <= 0.157:
        penalty = 1e3  # Large penalty
        difference += penalty
    """
    
    # Update best parameters if current difference is smaller
    if difference < best_difference:
        best_difference = difference
        best_params = {
            'num_layers': num_layers,
            'nodes_per_layer': [n for n in nodes_per_layer if n > 0],
            'omnifold_iterations': omnifold_iterations
        }
        best_weights = weights_last.copy()
        print("Found new best parameters!")

    tf.keras.backend.clear_session()
    gc.collect()

    # Wait for 3 seconds to allow VRAM to reduce
    time.sleep(10)

    return difference

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


# Output the best parameters
print('Best parameters that minimize the difference:')
print(best_params)
