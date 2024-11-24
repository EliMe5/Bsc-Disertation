import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# To allow LaTex in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =====================================================
# Editable Parameters
# =====================================================

n_dim = 2                  # Number of dimensions
n_events = 10**5           # Number of events to generate
n_clusters = 6             # Number of clusters (Gaussian components)

phase_space_min = 5        # Minimum boundary of each dimension
phase_space_max = 25       # Maximum boundary of each dimension

standard_min = 0.25        # Minimum standard deviation of gaussian events
standard_max = 0.75        # Maximum standard deviation of gaussian events

smearing_std = 0.0         # Constant smearing standard deviation applied to all points
smearing_coef = 0.05       # Linear smearing factor applied to all points

point_num = 4000           # How many events will be generated in the real distribution
random = 42                # Set the random state of the distributions for reproducibility

show_plot = True           # If true plot will show
save_data = False          # If true will save data with apropriate names in current folder

central_spike = False      # Set to true to add a spike, false for no spike

# If plot is shown
max_points = 4000          # Amount of events plotted
visible_dim = 2            # Number of dimensions of the output plot

# Add custom background function
def background_function(data, min=phase_space_min, max=phase_space_max):
    return np.abs(np.prod((data - min + 0.4 ), axis=1) ** -1)

# =====================================================
# Helper Functions
# =====================================================

def generate_synthetic_atlas_data_with_smearing(
    n_dim, n_events, n_clusters, phase_space_min, phase_space_max, 
    std_min, std_max, smearing_std, smearing_coef, max_points, random_state=42):
    """
    Main function to generate synthetic ATLAS-like data with smearing effects.
    """
    rng = np.random.default_rng(random_state)
    
    if central_spike:
        gmm = initialize_gmm_with_spike(n_clusters, n_dim, phase_space_min, phase_space_max, std_min, std_max, random_state)
    else:
        gmm = initialize_gmm(n_clusters, n_dim, phase_space_min, phase_space_max, std_min, std_max, random_state)
    data = generate_uniform_data(n_dim, n_events, phase_space_min, phase_space_max)
    
    truth_weights = calculate_weights(data, gmm, n_events)
    smeared_gmm = apply_smearing_to_gmm(gmm, data, smearing_std, smearing_coef)
    detector_weights = calculate_weights(data, smeared_gmm, n_events, True)
    
    real_data = sample_real_data(data, detector_weights, max_points, rng)
    
    return data, truth_weights, detector_weights, real_data

def initialize_gmm(n_clusters, n_dim, phase_space_min, phase_space_max, std_min, std_max, random_state):
    """
    Initializes a Gaussian Mixture Model (GMM) with specified parameters.
    """
    rng = np.random.default_rng(random_state)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state)
    
    means = rng.uniform(phase_space_min, phase_space_max, (n_clusters, n_dim))
    covariances = []
    for _ in range(n_clusters):
        stds = rng.uniform(std_min, std_max, n_dim)
        U, _, _ = np.linalg.svd(rng.normal(size=(n_dim, n_dim)))
        covariance = U @ np.diag(stds ** 2) @ U.T
        covariances.append(covariance)
    covariances = np.array(covariances)
    
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = np.ones(n_clusters) / n_clusters
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    
    return gmm

def initialize_gmm_with_spike(n_clusters, n_dim, phase_space_min, phase_space_max, std_min, std_max, random_state):
    """
    Initializes a Gaussian Mixture Model (GMM) with specified parameters,
    and adds an extra Gaussian centered.
    """
    rng = np.random.default_rng(random_state)
    n_components = n_clusters + 1
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    
    # Initialize means and covariances for the main clusters
    means = rng.uniform(phase_space_min, phase_space_max, (n_clusters, n_dim))
    covariances = np.zeros((n_clusters, n_dim, n_dim))
    for i in range(n_clusters):
        stds = rng.uniform(std_min, std_max, n_dim)
        U, _, _ = np.linalg.svd(rng.normal(size=(n_dim, n_dim)))
        covariance = U @ np.diag(stds ** 2) @ U.T
        covariances[i] = covariance
    
    # Add the extra Gaussian
    coor = (phase_space_min + phase_space_max) / 2
    extra_mean = np.array([[coor, coor]])  # shape (1, n_dim)
    extra_stds = rng.uniform(std_min, std_max, n_dim)
    U, _, _ = np.linalg.svd(rng.normal(size=(n_dim, n_dim)))
    extra_covariance = U @ np.diag(extra_stds ** 2) @ U.T
    extra_covariance = extra_covariance[np.newaxis, :, :]  # Reshape to (1, n_dim, n_dim)
    
    # Append the extra Gaussian to the parameters
    means = np.vstack([means, extra_mean])
    covariances = np.concatenate([covariances, extra_covariance], axis=0)
    
    # Initialize weights
    weights = np.ones(n_components) / n_components  # Uniform weights
    
    # Update the GMM attributes
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
    
    return gmm

def generate_uniform_data(n_dim, n_events, phase_space_min, phase_space_max):
    """
    Generates uniform synthetic data over the phase space.
    """
    rng = np.random.default_rng(21)
    return rng.uniform(phase_space_min, phase_space_max, (n_events, n_dim))

def calculate_weights(data, gmm, n_events, background=False):
    """
    Calculates the weights based on the Gaussian Mixture Model (GMM).
    """
    probabilities = np.exp(gmm.score_samples(data))
    if background:
        unormalised_min_weight = background_function(data)
        min_weight = unormalised_min_weight * ( np.sum(probabilities) / np.sum(unormalised_min_weight) )
    else:
        min_weight = 1 / (2 * n_events)
    weights = np.maximum(probabilities, min_weight)
    weights /= np.sum(weights)
    return weights

def apply_smearing_to_gmm(gmm, data, smearing_std, smearing_coef):
    """
    Applies smearing effects to the GMM covariances based on data.
    """
    smearing_variances = (smearing_std + smearing_coef * np.abs(data)) ** 2
    smeared_gmm = GaussianMixture(n_components=gmm.n_components, covariance_type='full', random_state=gmm.random_state)
    
    smeared_covariances = gmm.covariances_.copy()
    for k in range(gmm.n_components):
        average_smearing_variance = np.mean(smearing_variances, axis=0)
        smeared_covariances[k] += np.diag(average_smearing_variance)
    
    smeared_gmm.means_ = gmm.means_
    smeared_gmm.covariances_ = smeared_covariances
    smeared_gmm.weights_ = gmm.weights_
    smeared_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(smeared_covariances))
    
    return smeared_gmm

def sample_real_data(data, distribution, max_points, rng):
    """
    Samples data points from the smeared Gaussian Mixture Model (GMM).
    """
    sampled_indices = rng.choice(data.shape[0], size=max_points, p=distribution, replace=True)
    real_data = data[sampled_indices]
    return real_data

def probabilistic_filter_points(sample, weights, max_points, skewe=30, return_indices=False):
    # Step 1: Invert and normalize weights to use as probabilities
    inverted_weights = (-1 / (np.exp((skewe / weights.max()) * (weights - weights.max() / 4)) + 1)) + 1
    inverted_weights = 1 - inverted_weights
    inverted_weights = inverted_weights / inverted_weights.sum()

    # Step 2: Initialize indices to keep track of samples
    indices = np.arange(len(sample))

    # Select random indices to delete based on the probability distribution
    delete_indices = np.random.choice(indices, size=len(sample)-max_points, p=inverted_weights, replace=False)

    # Determine the indices to keep
    keep_indices = np.delete(indices, delete_indices)

    # Filter the sample and weights
    sample_reduced = sample[keep_indices]
    weights_reduced = weights[keep_indices]

    if return_indices:
        return sample_reduced, weights_reduced, keep_indices
    else:
        return sample_reduced, weights_reduced

def apply_smearing(data, smearing_std, smearing_coef, random_state=42):
    rng = np.random.default_rng(random_state)
    
    # Calculate the standard deviation for smearing for each data point
    # The smearing is applied per dimension
    sigma = smearing_std + smearing_coef * np.abs(data)
    
    # Generate Gaussian noise with calculated sigma
    noise = rng.normal(loc=0, scale=sigma)
    
    # Add the noise to the original data
    data_smeared = data + noise
    
    return data_smeared

def add_noise_to_weights(points, weights):
    # Step 1: Calculate the noise function at each point
    noise = np.abs(background_function(points))

    # Step 3: Normalize the noise distribution
    noise /= np.sum(noise)
    
    # Step 4: Keep the maximum value between noise and weights for each element
    updated_weights = np.maximum(weights, noise)
    
    # Step 5: Normalize the new weight distribution
    updated_weights /= np.sum(updated_weights)
    
    return updated_weights

# =====================================================
# Generate Synthetic Data (Monte Carlo Simulation)
# =====================================================

# Generate synthetic data at the truth level
synthetic_truth, weights_truth, weights_detector, data_real = \
    generate_synthetic_atlas_data_with_smearing(
        n_dim, n_events, n_clusters, phase_space_min, phase_space_max,
        standard_min, standard_max, smearing_std, smearing_coef,
        point_num, random_state=random
    )

# =====================================================
# Visualize Data (Optional)
# =====================================================
try:
    unfolded_weights = np.load('unfolded_weights.npy')
except FileNotFoundError:
    print("No unfolded_weights file found")
    unfolded_weights = None

if show_plot:
    if n_dim > 3:
        # Reduce data for high-dimensional correlation
        synthetic_truth_reduced, weights_truth_reduced, indices_truth = probabilistic_filter_points(
            synthetic_truth, weights_truth, max_points, return_indices=True)
        synthetic_truth_reduced, weights_detector_reduced, indices_detector = probabilistic_filter_points(
            synthetic_truth, weights_detector, max_points, return_indices=True)
        data_real_reduced, _, indices_real = probabilistic_filter_points(
            data_real, np.ones(len(data_real)), max_points, return_indices=True)

        # Combine datasets for t-SNE
        combined_data = np.vstack((synthetic_truth_reduced, synthetic_truth_reduced, data_real_reduced))
        tsne = TSNE(n_components=visible_dim, random_state=0)
        data_reduced = tsne.fit_transform(combined_data)

        # Separate the reduced data
        num_truth = len(synthetic_truth_reduced)
        num_detector = len(synthetic_truth_reduced)
        data_reduced_truth = data_reduced[:num_truth]
        data_reduced_detector = data_reduced[num_truth:num_truth + num_detector]
        data_reduced_real = data_reduced[num_truth + num_detector:]

        # Prepare point sizes
        point_sizes_truth = weights_truth_reduced * n_events
        point_sizes_detector = weights_detector_reduced * n_events
        point_sizes_real = np.full(len(data_reduced_real), 20)

        # Prepare unfolded data if available
        if unfolded_weights is not None:
            unfolded_w = np.min(weights_truth) + (unfolded_weights[-1, 1, :] - np.min(unfolded_weights[-1, 1, :])) / (np.max(unfolded_weights[-1, 1, :]) - np.min(unfolded_weights[-1, 1, :])) * (np.max(weights_truth) - np.min(weights_truth))
            unfolded_w_reduced = unfolded_w[indices_truth]
            data_reduced_unfolded = data_reduced_truth
            point_sizes_unfolded = unfolded_w_reduced * n_events
        else:
            data_reduced_unfolded = None

        # Create 2x2 grid of subplots
        fig = plt.figure(figsize=(12, 12))

        if visible_dim == 2:
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            ax1.scatter(
                data_reduced_truth[:, 0], data_reduced_truth[:, 1],
                c='blue', marker='.', s=point_sizes_truth, alpha=0.5
            )
            ax1.set_title(r"Truth synthetic distribution, $ \mathbf T_0 $")
            ax1.set_xlabel("t-SNE Component 1")
            ax1.set_ylabel("t-SNE Component 2")

            ax2.scatter(
                data_reduced_detector[:, 0], data_reduced_detector[:, 1],
                c='red', marker='.', s=point_sizes_detector, alpha=0.5
            )
            ax2.set_title(r"Detector synthetic distribution, $ \mathbf D_0 $")
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")

            ax3.scatter(
                data_reduced_real[:, 0], data_reduced_real[:, 1],
                c='green', marker='.', s=point_sizes_real, alpha=0.5
            )
            ax3.set_title(r"Truth real distribution, $ \mathbf R $")
            ax3.set_xlabel("t-SNE Component 1")
            ax3.set_ylabel("t-SNE Component 2")

            if unfolded_weights is not None:
                ax4.scatter(
                    data_reduced_unfolded[:, 0], data_reduced_unfolded[:, 1],
                    c='blue', marker='.', s=point_sizes_unfolded, alpha=0.5
                )
                ax4.set_title(r"Unfolded distribution, $ \mathbf U $")
                ax4.set_xlabel("t-SNE Component 1")
                ax4.set_ylabel("t-SNE Component 2")
            else:
                ax4.axis('off')
                ax4.set_title("Unfolded distribution not available")
        elif visible_dim == 3:
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')

            ax1.scatter(
                data_reduced_truth[:, 0], data_reduced_truth[:, 1], data_reduced_truth[:, 2],
                c='blue', marker='.', s=point_sizes_truth, alpha=0.5
            )
            ax1.set_title(r"Truth synthetic distribution, $ \mathbf T_0 $")
            ax1.set_xlabel("t-SNE Component 1")
            ax1.set_ylabel("t-SNE Component 2")
            ax1.set_zlabel("t-SNE Component 3")

            ax2.scatter(
                data_reduced_detector[:, 0], data_reduced_detector[:, 1], data_reduced_detector[:, 2],
                c='red', marker='.', s=point_sizes_detector, alpha=0.5
            )
            ax2.set_title(r"Detector synthetic distribution, $ \mathbf D_0 $")
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")
            ax2.set_zlabel("t-SNE Component 3")

            ax3.scatter(
                data_reduced_real[:, 0], data_reduced_real[:, 1], data_reduced_real[:, 2],
                c='green', marker='.', s=point_sizes_real, alpha=0.5
            )
            ax3.set_title(r"Truth real distribution, $ \mathbf R $")
            ax3.set_xlabel("t-SNE Component 1")
            ax3.set_ylabel("t-SNE Component 2")
            ax3.set_zlabel("t-SNE Component 3")

            if unfolded_weights is not None:
                ax4.scatter(
                    data_reduced_unfolded[:, 0], data_reduced_unfolded[:, 1], data_reduced_unfolded[:, 2],
                    c='blue', marker='.', s=point_sizes_unfolded, alpha=0.5
                )
                ax4.set_title(r"Unfolded distribution, $ \mathbf U $")
                ax4.set_xlabel("t-SNE Component 1")
                ax4.set_ylabel("t-SNE Component 2")
                ax4.set_zlabel("t-SNE Component 3")
            else:
                ax4.axis('off')
                ax4.set_title("Unfolded distribution not available")
        else:
            print("visible_dim must be 2 or 3")
            exit()

        plt.tight_layout()
        plt.show()

    elif n_dim == 3:
        # Create a figure with a 2x2 grid of 3D subplots
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')

        # Plot synthetic_truth
        ax1.scatter(
            synthetic_truth[:, 0], synthetic_truth[:, 1], synthetic_truth[:, 2],
            c='blue', marker='.', s=weights_truth * n_events, alpha=0.5
        )
        ax1.set_title(r"Truth synthetic distribution, $ \mathbf T_0 $")
        ax1.set_xlabel("Observable 1")
        ax1.set_ylabel("Observable 2")
        ax1.set_zlabel("Observable 3")

        # Plot synthetic_truth
        ax2.scatter(
            synthetic_truth[:, 0], synthetic_truth[:, 1], synthetic_truth[:, 2],
            c='red', marker='.', s=weights_detector * n_events, alpha=0.5
        )
        ax2.set_title(r"Detector synthetic distribution, $ \mathbf D_0 $")
        ax2.set_xlabel("Observable 1")
        ax2.set_ylabel("Observable 2")
        ax2.set_zlabel("Observable 3")

        # Plot data_real
        ax3.scatter(
            data_real[:, 0], data_real[:, 1], data_real[:, 2],
            c='green', marker='.', s=20, alpha=0.5
        )
        ax3.set_title(r"Truth real distribution, $ \mathbf R $")
        ax3.set_xlabel("Observable 1")
        ax3.set_ylabel("Observable 2")
        ax3.set_zlabel("Observable 3")

        # Plot unfolded distribution if unfolded_weights is available
        if unfolded_weights is not None:
            unfolded_w = np.min(weights_truth) + (unfolded_weights[-1, 1, :] - np.min(unfolded_weights[-1, 1, :])) / (np.max(unfolded_weights[-1, 1, :]) - np.min(unfolded_weights[-1, 1, :])) * (np.max(weights_truth) - np.min(weights_truth))
            ax4.scatter(
                synthetic_truth[:, 0], synthetic_truth[:, 1], synthetic_truth[:, 2],
                c='blue', marker='.', s=unfolded_w * n_events, alpha=0.5
            )
            ax4.set_title(r"Unfolded distribution, $ \mathbf U $")
            ax4.set_xlabel("Observable 1")
            ax4.set_ylabel("Observable 2")
            ax4.set_zlabel("Observable 3")
        else:
            ax4.axis('off')
            ax4.set_title("Unfolded distribution not available")

        plt.tight_layout()
        plt.show()

    elif n_dim == 2:
        # Create a figure with a 2x2 grid of subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

        ax1.scatter(
            synthetic_truth[:, 0], synthetic_truth[:, 1],
            c='blue', marker='.', s=weights_truth * n_events, alpha=0.5
        )
        ax1.set_title(r"Truth synthetic distribution, $ \mathbf T_0 $")
        ax1.set_xlabel("Observable 1")
        ax1.set_ylabel("Observable 2")

        ax2.scatter(
            synthetic_truth[:, 0], synthetic_truth[:, 1],
            c='red', marker='.', s=weights_detector * n_events, alpha=0.5
        )
        ax2.set_title(r"Detector synthetic distribution, $ \mathbf D_0 $")
        ax2.set_xlabel("Observable 1")
        ax2.set_ylabel("Observable 2")

        ax3.scatter(
            data_real[:, 0], data_real[:, 1],
            c='green', marker='.', s=20, alpha=0.5  # Uniform point size
        )
        ax3.set_title(r"Truth real distribution, $ \mathbf R $")
        ax3.set_xlabel("Observable 1")
        ax3.set_ylabel("Observable 2")

        if unfolded_weights is not None:
            unfolded_w = np.min(weights_truth) + (unfolded_weights[-1, 1, :] - np.min(unfolded_weights[-1, 1, :])) / (np.max(unfolded_weights[-1, 1, :]) - np.min(unfolded_weights[-1, 1, :])) * (np.max(weights_truth) - np.min(weights_truth))
            ax4.scatter(
                synthetic_truth[:, 0], synthetic_truth[:, 1],
                c='blue', marker='.', s=unfolded_w * n_events, alpha=0.5
            )
            ax4.set_title(r"Unfolded distribution, $ \mathbf U $")
            ax4.set_xlabel("Observable 1")
            ax4.set_ylabel("Observable 2")
        else:
            ax4.axis('off')
            ax4.set_title("Unfolded distribution not available")

        plt.tight_layout()
        plt.show()

# =====================================================
# Export Data for External Use (Optional)
# =====================================================

if save_data:
    # Save synthetic truth
    np.save('theta0_S.npy', synthetic_truth)
    np.save('initial_weights.npy', weights_truth)

    # Save synthetic detector
    np.save('smeared_weights.npy', weights_detector)

    # Save the real data
    np.save('theta_unknown_S.npy', data_real)
