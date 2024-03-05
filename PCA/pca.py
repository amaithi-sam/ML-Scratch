import numpy as np 
 
 
class PCA:
     
    def __init__(self, n_components):
        self.n_components = n_components 
        self.components = None 
        self.mean = None 

    def fit(self, X):
        
        # mean 
        self.mean = np.mean(X, axis=0)
        
        X = X - self.mean
        
        # Covariance
            # row = 1 sample, columns = feature
        cov = np.cov(X.T) # Transpose
        
        # EigenVectors, EigenValues 
        eigenvalues, eigenvectors = np.linalg.eig(cov)
            # v[:, i]
        # Sort EigenVectors 
        eigenvectors = eigenvectors.T 
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Store first n eigenVectors
        self.components = eigenvectors[0:self.n_components]
    
    def transform(self, X):
        
        # Project the data 
        X = X -self.mean 
        return np.dot(X, self.components.T)
        