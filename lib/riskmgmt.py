# Library for risk management
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.tsa.arima.model import ARIMA

## 1. Covariance estimation techniques
def compute_ew_cov_matrix(data, _lambda):
    if isinstance(data, pd.DataFrame): data = data.to_numpy() # convert pandas dataframe to numpy array when necessary
    # first compute the normalized weights (W_{t-n}, ..., W_{t-1}) for a certain lambda
    n = data.shape[0]
    weights = np.zeros(n)
    for i in range(1, n+1): weights[i-1] = (1-_lambda)* (_lambda ** (i-1))
    weights = (weights/sum(weights))[::-1]

    # then compute the covariance matrix with weights
    data_norm = (data-np.mean(data)) # (X_{t-i}-\mu_x) from t-n to t-1 (topmost row is t-n, bottommost row is t-1)
    weights = np.diag(weights)
    return ((weights@data_norm).T)@data_norm

## 2. Non PSD fixes for correlation matrices
### Cholesky 
def chol_psd(a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root = np.zeros((n, n))

    # loop over columns
    for j in range(n):
        s = 0.0
        # if we are not on the first column, calculate the dot product of the preceeding row values.
        if j > 0: s = root[j, :j].T@root[j, :j]
        # Diagonal Element
        temp = a[j, j]-s
        if -1e-8 <= temp <= 0: temp = 0.0
        root[j, j] = np.sqrt(temp);

        # Check for the 0 eigen value.  Just set the column to 0 if we have one
        if root[j, j] == 0.0:
            root[j, (j+1):n] = 0.0
        else:
            # update off diagonal rows of the column
            ir = 1.0/root[j, j]
            for i in range(j+1, n):
                s = root[i, :j].T@root[j, :j]
                root[i, j] = (a[i, j]-s) * ir 
    return root

### Near psd matrix
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1.0/np.sqrt(np.diag(out)))
        out = invSD@out@invSD

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (np.square(vecs)@vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T@vecs@l
    out = B@(B.T)

    # Add back the variance
    if invSD is not None: 
        invSD = np.diag(1.0/np.diag(invSD))
        out = invSD@out@invSD
    return out

### Higham’s 2002 nearest psd correlation function.
def proj_u(a):
    np.fill_diagonal(a, 1.0)
    return a

def proj_s(a, epsilon, weight):
    a = np.sqrt(weight)@a@np.sqrt(weight)
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    return np.sqrt(weight)@vecs@np.diag(vals)@(vecs.T)@np.sqrt(weight)

def Frobenius_norm(a, b):
    return np.sqrt(np.sum((a-b)**2))

def higham_near_psd(a, epsilon=1e-8, tol=1e-8, max_iter=1000, weight=None):
    n = a.shape[0]
    if weight is None: weight = np.identity(n)

    invSD = None
    gamma = np.inf
    y = a.copy()
    y0 = a.copy()
    ds = np.zeros_like(y)

    if np.count_nonzero(np.diag(y) == 1.0) != n:
        invSD = np.diag(1.0/np.sqrt(np.diag(out)))
        out = invSD@out@invSD

    for _ in range(max_iter):
        r = y-ds
        x = proj_s(r, epsilon, weight)
        y = proj_u(x)
        norm = Frobenius_norm(y, y0)
        minEigVal = np.real(np.linalg.eigvals(y)).min()
        if abs(norm-gamma) < tol and minEigVal > -epsilon:
            break
        gamma = norm
    
    if invSD is not None:
        invSD = np.diag(1.0/np.diag(invSD))
        y = invSD@y@invSD

    return y

### Confirm the matrix is now PSD.
def is_psd(a):
    return np.all(np.linalg.eigvals(a) >= 0)

## 3. Simulation Methods
### A multivariate normal simulation that allows for simulation directly from a covariance matrix or using PCA with an optional parameter for % variance explained.
def multi_norm_sim(cov_matrix, num_samples=10000, mean=0, var_explained=1.0, sim='direct'):
    if sim == 'direct':
        L = chol_psd(cov_matrix)
        normal_samples = np.random.normal(size=(cov_matrix.shape[0], num_samples))
        samples = (L@normal_samples).T+mean
        return samples
    elif sim == 'PCA':
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # filter out non-positive eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 0]
        eigenvectors = eigenvectors[:, eigenvalues > 0]

        # sort the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices] 
        eigenvectors = eigenvectors[:, sorted_indices]
        if var_explained == 1.0:
            var_explained = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]
        
        # find the number of principal components that explains the desired variance
        num_pc = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= var_explained)[0][0]+1
        eigenvectors = eigenvectors[:,:num_pc]
        eigenvalues = eigenvalues[:num_pc]

        # simulate the samples
        normal_samples = np.random.normal(size=(num_pc, num_samples))
        L = eigenvectors@np.diag(np.sqrt(eigenvalues))
        samples = (L@normal_samples).T+mean
        return samples
    else:
        raise Exception("sim must be either 'direct' or 'PCA'")

## 4. Value at Risk (VaR) calculation methods (all discussed)
def compute_VaR(data, alpha=0.05, mean=0):
    return mean-np.quantile(data, alpha)

### Using a normal distribution
def calculate_VaR_normal(returns, mean=0, alpha=0.05, num_samples=1000):
    mean = np.mean(returns)
    std = np.std(returns)
    norm_dist = np.random.normal(mean, std, num_samples)
    VaR = compute_VaR(norm_dist, alpha, mean)
    return VaR

### Using a normal distribution with an Exponentially Weighted variance
def calculate_VaR_normal_ew(returns, _lambda=0.94, alpha=0.05, mean=0, num_samples=1000):
    mean = np.mean(returns)
    std = np.sqrt(compute_ew_cov_matrix(returns, _lambda=_lambda))
    norm_dist_ew = np.random.normal(mean, std, num_samples)
    VaR = compute_VaR(norm_dist_ew, alpha, mean)
    return VaR

### Using a MLE fitted T distribution
def calculate_VaR_t_MLE(returns, alpha=0.05, mean=0, num_samples=1000):
    result = t.fit(returns, method="MLE")
    df, loc, scale = result
    t_dist = t(df, loc, scale).rvs(num_samples)
    VaR = compute_VaR(t_dist, alpha, mean)
    return VaR

### Using a fitted AR(1) model
def calculate_VaR_AR1(returns, alpha=0.05, mean=0, num_samples=1000):
    mean = np.mean(returns)
    ar1 = ARIMA(returns, order=(1, 0, 0))
    fitted = ar1.fit()
    std = np.sqrt(fitted.params['sigma2'])
    norm_dist = np.random.normal(0, std, num_samples)
    VaR = compute_VaR(norm_dist, alpha, mean)
    return VaR

### Using a Historic Simulation
def calculate_VaR_historic(returns, alpha=0.05, mean=0):
    VaR = compute_VaR(returns, alpha, mean)
    return VaR

## 5. Expected Shortfall (ES) calculation
def compute_ES(returns, alpha=0.05, mean=0):
    VaR = compute_VaR(returns, alpha, mean)
    ES = -np.mean(returns[returns <= -VaR])
    return ES

## 6. Other functions 
### A function that allow the user to specify the method of return calculation.
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars = prices.columns
    nVars = len(vars)
    vars = vars[vars != dateColumn]
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars-1

    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2-1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[dateColumn].iloc[1:n].to_numpy()
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:, i]

    return out
