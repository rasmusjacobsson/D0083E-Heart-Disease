from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Run PCA to get coeffs, score, latent, tsquared, explained, mu
def run_pca(x, n_components=None):
    # Save mean and std for later reconstruction
    std = np.std(x, axis=0)
    mean = np.mean(x, axis=0)
    x_std = (x - mean) / std
    # PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x_std)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    pca_mean = pca.mean_  # mean of standardized data (should be ~0)
    return {
        "coeffs": components,
        "score": principal_components,
        "latent": explained_variance,
        "tsquared": np.sum((principal_components / np.std(principal_components, axis=0))**2, axis=1),
        "explained": explained_variance_ratio,
        "mu": mean,
        "std": std
    }

# Plot scree plot
def plot_scree(latent):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(latent) + 1), latent, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, len(latent) + 1))
    plt.grid()
    plt.tight_layout()

# Plot explained variance
def plot_explained_variance(explained_variance_ratio):
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
    plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance by Principal Components')
    plt.legend(loc='best')
    plt.tight_layout()

# Plot biplot
def plot_biplot(score, coeffs, feature_names):
    plt.figure(figsize=(8, 8))
    plt.scatter(score[:, 0], score[:, 1], alpha=0.5)
    for i in range(coeffs.shape[1]):
        plt.arrow(0, 0, coeffs[0, i]*max(score[:, 0]), coeffs[1, i]*max(score[:, 1]), color='r', alpha=0.5)
        plt.text(coeffs[0, i]*max(score[:, 0])*1.15, coeffs[1, i]*max(score[:, 1])*1.15, feature_names[i], color='g', ha='center', va='center')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    plt.grid()
    plt.tight_layout()

# 3D PCA scores plot
def plot_3D_scores(score):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(score[:, 0], score[:, 1], score[:, 2], alpha=0.5)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Scores Plot')
    plt.tight_layout()

# Determine number of components to reach a certain variance threshold and kaiser criterion
def k_components(explained_variance_ratio, threshold=0.95):
    cumulative_variance = np.cumsum(explained_variance_ratio)
    k_components_threshold = np.argmax(cumulative_variance >= threshold) + 1
    k_components_kaiser = np.sum(explained_variance_ratio > (1 / len(explained_variance_ratio)))
    return k_components_threshold, k_components_kaiser

# Plot top 5 PCA loadings as a heatmap
def plot_loadings(coeffs, feature_names):
    # Find the top 5 contributors to the first 5 principal components
    top_features = {}
    for i in range(5):
        pc_loadings = coeffs[i]
        top_indices = np.argsort(np.abs(pc_loadings))[-5:][::-1]
        top_features[f'PC {i+1}'] = [(feature_names[idx], np.abs(pc_loadings[idx])) for idx in top_indices]
    # print top features as a table
    print("Top 5 features contributing to the first 5 principal components:")
    print("{:<10} {:<30} {:<10}".format("PC", "Feature", "Loading"))
    for pc, features in top_features.items():
        for feature, loading in features:
            print("{:<10} {:<30} {:<10.2f}".format(pc, feature, loading))

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.abs(coeffs[:5].T), annot=True, fmt=".2f", cmap="coolwarm", yticklabels=feature_names, xticklabels=[f'PC {i+1}' for i in range(5)])
    plt.title('Top 5 PCA Loadings')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()

# Reconstruct original data and calculate rmse
def pca_reconstruction(X, score, coeffs, mu, std, n_components):
    # Ensure mu and std are numpy arrays for broadcasting
    mu = np.asarray(mu)
    std = np.asarray(std)
    # Reconstruct the standardized data
    reconstructed_std = np.dot(score[:, :n_components], coeffs[:n_components, :])
    # Unstandardize using provided mean and std
    reconstructed = reconstructed_std * std + mu
    # Calculate RMSE using original data X
    rmse = np.sqrt(np.mean((reconstructed - X) ** 2))
    return reconstructed, rmse
