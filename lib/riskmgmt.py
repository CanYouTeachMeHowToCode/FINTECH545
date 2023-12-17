# Library for risk management
import numpy as np
import pandas as pd
from scipy.stats import t, norm
from scipy.optimize import minimize
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

### A function that compute the pairwise covariance matrix with input data that includes some NA values
def pairwise_cov(x, skipMiss=True, func=np.cov):
    n, m = x.shape
    nMiss = np.sum(np.isnan(x), axis=0)

    # nothing missing, just calculate it.
    if np.sum(nMiss) == 0:
        return func(x)

    idxMissing = [set(np.where(np.isnan(x[:, i]))[0]) for i in range(m)]

    if skipMiss:
        # Skipping Missing, get all the rows which have values and calculate the covariance
        rows = set(range(n))
        for c in range(m):
            for rm in idxMissing[c]:
                if rm in rows:
                    rows.remove(rm)
        rows = sorted(list(rows))
        return func(x[rows,:].T)

    else:
        # Pairwise, for each cell, calculate the covariance.
        out = np.empty((m, m))
        for i in range(m):
            for j in range(i+1):
                rows = set(range(n))
                for c in (i,j):
                    for rm in idxMissing[c]:
                        if rm in rows:
                            rows.remove(rm)
                rows = sorted(list(rows))
                out[i,j] = func(x[rows,:][:,[i,j]].T)[0,1]
                if i != j:
                    out[j,i] = out[i,j]
        return out

### function to compute the Super Efficient, Maximum Sharpe Ratio, portfolio.
def max_sharpe_ratio_portfolio(ER, cov, rf):
    func = lambda wts: -(wts@ER-rf)/np.sqrt(wts@cov@wts)
    x0 = np.full(ER.shape[0], 1/ER.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x}, 
            {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(ER.shape[0])]
    res = minimize(func, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res

### function to compute the risk budgets (rist portion of the portfolio)
def risk_budget(wts, cov):
    portfolioStd = np.sqrt(wts@cov@wts)
    csd = wts*(cov@wts)/portfolioStd
    return csd/portfolioStd

### function to compute the risk parity portfolio
def risk_parity_portfolio(cov):
    func = lambda wts: (wts*(cov@wts)/np.sqrt(wts@cov@wts)).std()
    x0 = np.full(cov.shape[0], 1/cov.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
            {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(cov.shape[0])]
    res = minimize(func, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res

#############################################
### function to compute risk attributions
def risk_contrib(w, covar):
    risk_contrib = w * covar.dot(w) / np.sqrt(w.dot(covar).dot(w))
    return risk_contrib

def expost_attribution(w, upReturns):
    _stocks = list(upReturns.columns)
    n = upReturns.shape[0]
    pReturn = np.empty(n)
    weights = np.empty((n, len(w)))
    lastW = np.copy(w)
    matReturns = upReturns[_stocks].values

    for i in range(n):
        # Save Current Weights in Matrix
        weights[i,:] = lastW

        # Update Weights by return
        lastW = lastW * (1.0 + matReturns[i,:])

        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastW)
        # Normalize the wieghts back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1

    # Set the portfolio return in the Update Return DataFrame
    upReturns['Portfolio'] = pReturn

    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    # Calculate the Carino K
    k = np.log(totalRet + 1) / totalRet

    # Carino k_t is the ratio scaled by 1/K 
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(matReturns * (weights * carinoK[:, np.newaxis]), columns=_stocks)

    # Set up a Dataframe for output.
    Attribution = pd.DataFrame({'Value': ["TotalReturn", "Return Attribution"]})

    _ss = list(upReturns.columns)
    _ss.append('Portfolio')
    
    for s in _ss:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(upReturns[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        atr =  attrib[s].sum() if s != 'Portfolio' else tr
        # Set the values
        Attribution[s] = [tr, atr]

        # Y is our stock returns scaled by their weight at each time
        Y =  matReturns * weights
        # Set up X with the Portfolio Return
        X = np.column_stack((np.ones((pReturn.shape[0], 1)), pReturn))
        # Calculate the Beta and discard the intercept
        B = (np.linalg.inv(X.T @ X) @ X.T @ Y)[1,:]
        # Component SD is Beta times the standard Deviation of the portfolio
        cSD = B * np.std(pReturn)

        Expost_Attribution = pd.concat([Attribution,    
            pd.DataFrame({"Value": ["Vol Attribution"], 
                        **{_stocks[i]: [cSD[i]] for i in range(len(_stocks))},
                        "Portfolio": [np.std(pReturn)]})
        ], ignore_index=True)

    return Expost_Attribution

def expost_factor(w, upReturns, upFfData, Betas):
    stocks = upReturns.columns
    factors = list(upFfData.columns)
    
    n = upReturns.shape[0]
    m = len(stocks)
    
    pReturn = np.empty(n)
    residReturn = np.empty(n)
    weights = np.empty((n, len(w)))
    factorWeights = np.empty((n, len(factors)))
    lastW = w.copy()
    matReturns = upReturns[stocks].to_numpy()
    ffReturns = upFfData[factors].to_numpy()

    for i in range(n):
        # Save Current Weights in Matrix
        weights[i,:] = lastW

        #Factor Weight
        factorWeights[i,:] = Betas.T @ lastW

        # Update Weights by return
        lastW = lastW * (1.0 + matReturns[i,:])

        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastW)
        # Normalize the weights back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1

        # Residual
        residReturn[i] = (pR-1) - factorWeights[i,:] @ ffReturns[i,:]

    # Set the portfolio return in the Update Return DataFrame
    upFfData["Alpha"] = residReturn
    upFfData["Portfolio"] = pReturn

    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    # Calculate the Carino K
    k = np.log(totalRet + 1) / totalRet

    # Carino k_t is the ratio scaled by 1/K 
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(ffReturns * (factorWeights * carinoK[:, np.newaxis]), columns=factors)
    attrib["Alpha"] = residReturn * carinoK

    # Set up a DataFrame for output.
    Attribution = pd.DataFrame({"Value": ["TotalReturn", "Return Attribution"]})

    
    newFactors = factors[:]
    newFactors.append('Alpha')

    ss = factors[:]
    ss.append('Alpha')
    ss.append('Portfolio')

    # Loop over the factors
    for s in ss:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(upFfData[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        atr = sum(attrib[s]) if s != "Portfolio" else tr
        # Set the values
        Attribution[s] = [tr, atr]

    # Realized Volatility Attribution

    # Y is our stock returns scaled by their weight at each time
    Y = np.hstack((ffReturns * factorWeights, residReturn[:,np.newaxis]))
    # Set up X with the Portfolio Return
    X = np.hstack((np.ones((n,1)), pReturn[:,np.newaxis]))
    # Calculate the Beta and discard the intercept
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    B = B[1,:]
    # Component SD is Beta times the standard Deviation of the portfolio
    cSD = B * np.std(pReturn)

    # Check that the sum of component SD is equal to the portfolio SD
    assert np.isclose(np.sum(cSD), np.std(pReturn))

    # Add the Vol attribution to the output
    Expost_Attribution = pd.concat([Attribution, 
        pd.DataFrame({"Value": "Vol Attribution", **{newFactors[i]:cSD[i] for i in range(len(newFactors))}, "Portfolio":np.std(pReturn)}, index=[0])
    ])

    return Expost_Attribution
#############################################

### GBSM
class GBSM:
    def __init__(self, S, X, T, sigma, r, b, option_type):
        self.S = S
        self.X = X
        self.T = T
        self.r = r
        self.b = b
        self.sigma = sigma
        self.option_type = option_type
        self.d1 = (np.log(S/X)+(b+sigma**2/2)*T)/(sigma*np.sqrt(T))
        self.d2 = self.d1-sigma*np.sqrt(T)

    # Function for computing Black-Scholes prices (values)
    def black_scholes(self):
        if self.option_type == 'Call':
            return self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(self.d1)-self.X*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        elif self.option_type == 'Put':
            return self.X*np.exp(-self.r*self.T)*norm.cdf(-self.d2)-self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(-self.d1)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'")
    
    def _delta(self):
        if self.option_type == 'Call':
            return np.exp((self.b-self.r)*self.T)*norm.cdf(self.d1)
        elif self.option_type == 'Put':
            return np.exp((self.b-self.r)*self.T)*(norm.cdf(self.d1)-1)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'")
    
    def _gamma(self):
        return norm.pdf(self.d1)*np.exp((self.b-self.r)*self.T)/(self.S*self.sigma*np.sqrt(self.T))

    def _vega(self):
        return self.S*np.exp((self.b-self.r)*self.T)*norm.pdf(self.d1)*np.sqrt(self.T)

    def _theta(self):
        if self.option_type == 'Call':
            return -self.S*np.exp((self.b-self.r)*self.T)*norm.pdf(self.d1)*self.sigma/(2*np.sqrt(self.T))-(self.b-self.r)*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(self.d1)-self.r*self.X*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        elif self.option_type == 'Put':
            return -self.S*np.exp((self.b-self.r)*self.T)*norm.pdf(self.d1)*self.sigma/(2*np.sqrt(self.T))+(self.b-self.r)*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(-self.d1)+self.r*self.X*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'")

    def _rho(self): # Note: original formual assumes r=b, yet it does not hold in this case, so we redo the derivation
        if self.option_type == 'Call': # Call: rho = -T*S*exp((b-r)*T)*N(d1)+T*X*exp(-r*T)*N(d2)
            return -self.T*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(self.d1)+self.T*self.X*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        elif self.option_type == 'Put': # Put: rho = T*S*exp((b-r)*T)*N(-d1)-T*X*exp(-r*T)*N(-d2)
            return self.T*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(-self.d1)-self.T*self.X*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'")
        
    def _carry_rho(self):
        if self.option_type == 'Call':
            return self.T*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(self.d1)
        elif self.option_type == 'Put':
            return -self.T*self.S*np.exp((self.b-self.r)*self.T)*norm.cdf(-self.d1)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'")

    def getAllGreeks(self):
        return {'delta': self._delta(), 'gamma': self._gamma(), 'vega': self._vega(), 'theta': self._theta(), 'rho': self._rho(), 'carry_rho': self._carry_rho()}

