from scipy.spatial.distance import cdist
import numpy as np
import torch

class MMD:
    def __init__(self, X, Y, sigma=1.0):
        """
        X, Y: Feature matrices (samples × features)
        sigma: Bandwidth parameter for the RBF kernel
        """
        self.X = X
        self.Y = Y
        self.sigma = sigma
    
    def rbf_kernel(self, X, Y):
        """Compute the RBF kernel matrix."""
        pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-pairwise_sq_dists / (2 * self.sigma**2))
    
    def compute_mmd(self):
        """Compute the unbiased MMD estimate."""
        m, n = self.X.shape[0], self.Y.shape[0]
        k_xx = self.rbf_kernel(self.X, self.X)
        k_yy = self.rbf_kernel(self.Y, self.Y)
        k_xy = self.rbf_kernel(self.X, self.Y)
        
        term_xx = np.sum(k_xx) / (m * (m - 1))  # Expectation over P
        term_yy = np.sum(k_yy) / (n * (n - 1))  # Expectation over Q
        term_xy = np.sum(k_xy) / (m * n)  # Cross term
        
        mmd_score = term_xx + term_yy - 2 * term_xy
        return mmd_score

# Example usage
if __name__ == "__main__":
    # Generate random feature vectors
    X = np.random.normal(0, 1, (100, 512))  # Simulating CLIP features
    Y = np.random.normal(0, 1, (100, 512))
    
    mmd = MMD(X, Y, sigma=10)
    score = mmd.compute_mmd()
    print(f"MMD Score: {score}")
