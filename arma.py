#!/usr/bin/env python
# coding: utf-8

# In[78]:


from scipy.stats.distributions import norm
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from pydataset import data as pydata
from statsmodels.tsa.stattools import arma_order_select_ic as order_select
import pandas as pd
from scipy.linalg import block_diag

def kalman(F, Q, H, time_series):
    # Get dimensions
    dim_states = F.shape[0]

    # Initialize variables
    # covs[i] = P_{i | i-1}
    covs = np.zeros((len(time_series), dim_states, dim_states))
    mus = np.zeros((len(time_series), dim_states))

    # Solve of for first mu and cov
    covs[0] = np.linalg.solve(np.eye(dim_states**2) - np.kron(F,F),np.eye(dim_states**2)).dot(Q.flatten()).reshape(
            (dim_states,dim_states))
    mus[0] = np.zeros((dim_states,))

    # Update Kalman Filter
    for i in range(1, len(time_series)):
        t1 = np.linalg.solve(H.dot(covs[i-1]).dot(H.T),np.eye(H.shape[0]))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs


def state_space_rep(phis, thetas, mu, sigma):
    # Initialize variables
    dim_states = max(len(phis), len(thetas)+1)
    dim_time_series = 1 #hardcoded for 1d time_series

    F = np.zeros((dim_states,dim_states))
    Q = np.zeros((dim_states, dim_states))
    H = np.zeros((dim_time_series, dim_states))

    # Create F
    F[0][:len(phis)] = phis
    F[1:,:-1] = np.eye(dim_states - 1)
    # Create Q
    Q[0][0] = sigma**2
    # Create H
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas

    return F, Q, H, dim_states, dim_time_series


def arma_forecast_naive(file='weather.npy',p=2,q=1,n=20):
    """
    Perform ARMA(1,1) on data. Let error terms be drawn from
    a standard normal and let all constants be 1.
    Predict n values and plot original data with predictions.

    Parameters:
        file (str): data file
        p (int): order of autoregressive model
        q (int): order of moving average model
        n (int): number of future predictions
    """
    #Initialize the Data and take the difference of the Data
    weather = np.diff(np.load("weather.npy"))
    phi = 0.5*np.ones(p)
    theta = .1*np.ones(q)
    eps0 = np.random.normal(0,1)
    Z = weather[-p:].tolist()
    
    #Looping through all of the different time steps
    for i in range(n):
        eps = np.random.normal(0,1) 
        Z1 = np.array(Z[-p:]).T@phi.reshape(p,1) + eps + theta.dot(eps0)
        Z.append(Z1[0])
        eps0 = eps
    
    #plotting the results
    datetime_col = pd.date_range(start='05-13-2019t18:56', periods=71 + n, freq="1H")
    df = pd.DataFrame(index = datetime_col)
    plt.figure(figsize = (10,7))
    df["Weather"] = np.array(weather.tolist() + Z[2:])
    df["Weather"].iloc[:72].plot()
    df["Weather"].iloc[72:].plot()
    plt.title("Naive ARMA")
    plt.xlabel("Dates")
    plt.ylabel("Change in Temperature")
    plt.show()
    
    return


def arma_likelihood(file='weather.npy', phis=np.array([0]), thetas=np.array([0]), mu=0., std=1.):
    """
    Transfer the ARMA model into state space. 
    Return the log-likelihood of the ARMA model.

    Parameters:
        file (str): data file
        phis (ndarray): coefficients of autoregressive model
        thetas (ndarray): coefficients of moving average model
        mu (float): mean of errorm
        std (float): standard deviation of error

    Return:
        log_likelihood (float)
    """
    weather = np.diff(np.load("weather.npy"))
    n = len(weather)
    
    #Initial Setup of inputs
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    mus, covs = kalman(F, Q, H,  weather - mu)
    
    #Block Diagonal Speed up code
    try1 = (H@mus.T).reshape(n,)
    try2 = (block_diag(*[H]*n)@np.vstack(covs)@H.T).reshape(n,)
    
    return np.sum(np.log(norm(try1 + mu, np.sqrt(try2)).pdf(weather)))


def model_identification(file='weather.npy',p=4,q=4):
    """
    Identify parameters to minimize AIC of ARMA(p,q) model

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model

    Returns:
        phis (ndarray (p,)): coefficients for AR(p)
        thetas (ndarray (q,)): coefficients for MA(q)
        mu (float): mean of error
        std (float): std of error
    """
    import sys
    minp = 0
    minj = 0
    min_score = sys.maxsize
    weather = np.diff(np.load("weather.npy"))
    
    n = len(weather)
    
    from scipy.optimize import fmin
    # assume p, q, and time_series are defined
    for i in range(1,p+1):
        for j in range(1,q+1):
            def f(x): # x contains the phis, thetas, mu, and std
                return -1*arma_likelihood(weather, phis=x[:i], thetas=x[i:i+j], mu= x[-2], std=x[-1])
            
            # create initial point
            x0 = np.zeros(i+j+2)
            x0[-2] = weather.mean()
            x0[-1] = weather.std()
            sol = fmin(f,x0,maxiter=10000, maxfun=10000)
            
            k = i + j + 2
            
            def aic(x,k,n):
                return 2*k*(1 + (k+1)/(n-k)) + 2*f(x)
            
            check = aic(sol,k,n)
            if check < min_score:
                min_score = check
                minp = i
                minq = j
                minsol = sol
            
    return minsol[:minp],minsol[minp:minp+minq],minsol[-2],minsol[-1]


def arma_forecast(file='weather.npy', phis=np.array([0]), thetas=np.array([0]), mu=0., std=0., n=30):
    """
    Forecast future observations of data.

    Parameters:
        file (str): data file
        phis (ndarray (p,)): coefficients of AR(p)
        thetas (ndarray (q,)): coefficients of MA(q)
        mu (float): mean of ARMA model
        std (float): standard deviation of ARMA model
        n (int): number of forecast observations

    Returns:
        new_mus (ndarray (n,)): future means
        new_covs (ndarray (n,)): future standard deviations
    """
    z = np.load(file)
    z = z[1:] - z[:-1]
    l = len(z)

    # set up state space
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    xs, Ps = kalman(F, Q, H, z - mu)

    z = list(z)
    #Initial Update
    y = (z[-1] - mu) - H@xs[-1]
    S = H@Ps[-1]@H.T
    K = Ps[-1]@H.T@np.linalg.inv(S)
    x_k = xs[-1] + K@y
    P_k = (np.eye(len(K)) - K@H)@Ps[-1]
    
    #Single Update prediction function
    def predict(x, P):
        x_n = F@x
        P_n = F@P@F.T +Q
        return x_n, P_n

    new_xs = [x_k]
    new_Ps = [P_k]
    
    #Looping through all times that the update is needed for
    for i in range(int(n) + 1):
        x_p, P_p = predict(new_xs[-1], new_Ps[-1])
        new_xs.append(x_p)
        new_Ps.append(P_p)

    new_xs = new_xs[1:]
    
    #Plotting the results of the update
    datetime_col = pd.date_range(start='04-13-2019t19:56', periods=(l + n + 1), freq="1h")
    data = pd.DataFrame(z + [(H@x)[0] + mu for x in new_xs], index=datetime_col, columns=["weather"])
    data["weather"].iloc[:l].plot(label="Data",figsize = (15,10))
    data["weather"].iloc[l:].plot(label="Predicted")
    data['upper'] = data['weather'] + 2 * std
    data["upper"].iloc[l:].plot(label="95% Confident Interval Upper", color="r")
    data['lower'] = data['weather'] - 2 * std
    data["lower"].iloc[l:].plot(label="Lower Bound", color='r')
    
    plt.legend()
    plt.title("ARMA Forecast(1,1)")
    plt.show()

    return np.array([(H@x)[0] + mu for x in new_xs])[:-1], np.array([H@P@H.T for P in new_Ps]).flatten()[1:-1]


def sm_arma(file = 'weather.npy', p=3, q=3, n=30):
    """
    Build an ARMA model with statsmodel and 
    predict future n values.

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model
        n (int): number of values to predict

    Return:
        aic (float): aic of optimal model
    """
    #Initialize the data and parameters for the data
    z = np.diff(np.load(file))
    l = len(z)

    min_aic = np.inf
    bestp  = 0
    bestq = 0
    datetime_col = pd.date_range(start='04-13-2019t19:56', periods=(l), freq="1h")
    data = pd.DataFrame(z, index=datetime_col, columns=["weather"])
    
    #Idendifying the best model groups
    for i in range(1,p+1):
        for j in range(1,q+1):
            model = ARMA(z, order=(i, j))
            model = model.fit(method='mle', trend='c')
            pred = model.predict(start=0, end=(l + 30))

            aic = model.aic
            if aic < min_aic:
                min_aic = aic
                bestp = i
                bestq = j
                
    model = ARMA(z, order=(bestp, bestq))
    model = model.fit(method='mle', trend='c')
    pred = model.predict(start=0, end=(l + 30))
    
    #Plotting the best fit models and results
    plt.figure(figsize = (12,8))
    datetime_col = pd.date_range(start='04-13-2019t19:56', periods=(l + n+1), freq="1h")
    pred_df = pd.DataFrame(pred, index=datetime_col, columns=["weather"])
    data["weather"].plot(label="Data")
    pred_df["weather"].plot(label="Predicted")
    plt.title("Stats ARMA("+str(bestp)+","+str(bestq)+")")
    plt.ylabel("Change in Temperature")
    plt.xlabel("Dates")
    plt.legend()
    plt.show()
    
    return min_aic

def manaus(start='1983-01-31',end='1995-01-31',p=4,q=4):
    """
    Plot the ARMA(p,q) model of the River Negro height
    data using statsmodels built-in ARMA class.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
        p (int): max_ar parameter
        q (int): max_ma parameter
    Return:
        aic_min_order (tuple): optimal order based on AIC
        bic_min_order (tuple): optimal order based on BIC
    """
    # Get dataset
    raw = pydata('manaus')
    # Make DateTimeIndex
    manaus = pd.DataFrame(raw.values,index=pd.date_range('1903-01','1993-01',freq='M'))
    manaus = manaus.drop(0,axis=1)
    # Reset column names
    manaus.columns = ['Water Level']
    
    #Selecting the best order
    order = order_select(manaus.values, max_ar=p, max_ma=q, ic=['aic', 'bic'],fit_kw = {'method': 'mle'})
    aic = order['aic_min_order']
    bic = order['bic_min_order']
    
    #The Mle models
    model = ARMA(manaus, aic).fit(method='mle')
    fig, ax = plt.subplots(figsize=(13, 7))
    fig = model.plot_predict(start=start, end=end, ax=ax)
    
    #The Aic plot
    ax.set_title('Manaus Dataset AIC')
    ax.set_xlabel('Year')
    ax.set_ylabel('Water Level')
    plt.show()

    model = ARMA(manaus, bic).fit(method='mle')
    fig, ax = plt.subplots(figsize=(13, 7))
    fig = model.plot_predict(start=start, end=end, ax=ax)

    #The BIC plot
    ax.set_title('Manaus Dataset BIC')
    ax.set_xlabel('Year')
    ax.set_ylabel('Water Level')
    plt.show()

    return aic, bic

