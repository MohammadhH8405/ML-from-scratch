import numpy as np

class PCA :
    # A simple implementation of Principal Component Analysis (PCA).
    def __init__(self):
        '''
        Attributes :
        means : (np.array) The mean of each feature in the training data. Shape: (n_features,)
        eigen_vectors : (np.array) The principal component vectors. Shape: (n_features, k)
        '''
        self.means = None
        self.eigen_vectors = None

    def build(self, x, k) :
        """
        Fit PCA to the dataset and compute the top k principal components.
        Parameters :
        x : (np.array) Input data of shape (n_samples, n_features).
        k : (int) Number of principal components to keep. Must be < n_features.
        """
        if k >= x.shape[1] :
            raise ValueError(f"k={k} is too large — must be less than the number of features {x.shape[1]}.")
        # Step 1: Compute mean of each feature
        self.means = np.mean(x, axis=0)
        # Step 2: Center the data (subtract mean)
        x_normal = x - self.means
        # Step 3: Compute the covariance matrix (n_features x n_features)
        cov_matrix = (x_normal.T @ x_normal) / (len(x) - 1)
        # Step 4: Eigen decomposition of covariance matrix
        # eigh() for symmetrical matricis.
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Step 5: Select top k eigenvectors and store them
        eigen_pairs = [(eigenvalues[i], eigenvectors[:,i]) for i in range(len(eigenvalues))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eigen_vectors = np.array([i[1] for i in eigen_pairs[:k]])

    def reduce_dimensions(self, x):
        """
        Project x onto the learned principal components.
        Must call build() before using this method.
        """
        if self.eigen_vectors is None:
            raise RuntimeError("PCA not yet built — call build() first.")
        x_normal = x - self.means
        return x_normal @ self.eigen_vectors.T

    def increase_dimensions(self, x_proj):
        # Reconstruct an approximation of the original data
        return x_proj @ self.eigen_vectors + self.means
