import streamlit as st
import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import pandas as pd # Added the import of pandas

# --- Black Scholes Section Start ---
def black_scholes_option_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def calculate_implied_volatility(S, K, T, r, market_price, option_type="call", initial_guess=0.2):
    def price_difference(sigma):
        model_price = black_scholes_option_price(S, K, T, r, sigma, option_type)
        return model_price - market_price
    try:
        implied_vol = brentq(price_difference, 0.0001, 10, xtol=1e-8)
    except ValueError as e:
        print(f"Error finding implied volatility: {e}")
        return None
    return implied_vol
# --- Black Scholes Section End ---

def estimate_heston_parameters_mom(data_file):
    df = data_file
    prices = df['Close'].values
    log_returns = np.log(prices[1:] / prices[:-1])

    sample_mean = np.mean(log_returns)
    sample_var = np.var(log_returns)
    sample_skew = np.mean((log_returns - sample_mean)**3) / sample_var**(3/2)
    sample_kurt = np.mean((log_returns - sample_mean)**4) / sample_var**2

    def theoretical_moments(params):
        kappa, theta, vol_of_vol, rho = params
        v0 = sample_var
        t = 1/252

        theoretical_mean = sample_mean

        theoretical_var = v0 * (1 - np.exp(-kappa * t)) / kappa + theta * (t - (1 - np.exp(-kappa * t)) / kappa)

        theoretical_skew = (rho * vol_of_vol / np.sqrt(kappa)) * (v0 - theta) * (1 - np.exp(-kappa * t)) / np.sqrt(t)

        theoretical_kurt = 3 + 3 * vol_of_vol**2 / (kappa**2 * theta) * (1 - np.exp(-kappa*t) + (v0 - theta)/theta * (1 - np.exp(-kappa*t))**2)

        return theoretical_mean, theoretical_var, theoretical_skew, theoretical_kurt

    def objective_function(params):
        theoretical_mean, theoretical_var, theoretical_skew, theoretical_kurt = theoretical_moments(params)
        return (
            (sample_mean - theoretical_mean)**2 +
            (sample_var - theoretical_var)**2 +
            (sample_skew - theoretical_skew)**2 +
            (sample_kurt - theoretical_kurt)**2
        )

    initial_guess = [1.0, 0.04, 0.2, -0.7]
    bounds = [(0.001, None), (0.001, None), (0.001, None), (-1, 1)]
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)
    kappa, theta, vol_of_vol, rho = result.x
    return kappa, theta, vol_of_vol, rho

def estimate_drift_volatility(data_file):
    df = data_file
    prices = df['Close'].values
    log_returns = np.log(prices[1:] / prices[:-1])
    mu = np.mean(log_returns) * 252
    sigma = np.std(log_returns) * np.sqrt(252)
    return mu, sigma, prices[-1]

def simulate_price_path(S0, mu, sigma, T, dt, num_steps, lambda_jumps, jump_mean, jump_std, kappa, theta, vol_of_vol, rho):
    price_path = [S0]
    volatility_path = [sigma**2]
    current_price = S0
    current_volatility = sigma**2

    for step in range(num_steps):
        z1, z2 = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]])

        current_volatility = current_volatility + kappa * (theta - current_volatility) * dt + vol_of_vol * np.sqrt(current_volatility) * np.sqrt(dt) * z2
        current_volatility = max(current_volatility, 0.001)
        volatility_path.append(current_volatility)

        jump = 0
        if np.random.poisson(lambda_jumps * dt) > 0:
            jump = np.random.normal(jump_mean, jump_std)

        current_price = current_price * np.exp(
            (mu - 0.5 * current_volatility) * dt + np.sqrt(current_volatility) * np.sqrt(dt) * z1 + jump
        )
        price_path.append(current_price)
    return price_path, volatility_path

def monte_carlo_simulation(data_file, target_price, num_simulations, num_days,
                             lambda_jumps=1, jump_mean=0, jump_std=0.1,
                             kappa=1, theta=0.04, vol_of_vol=0.2, rho=-0.7,
                             option_type=None, strike=None, risk_free_rate=None):
    mu, sigma, S0 = estimate_drift_volatility(data_file)
    T = num_days / 252
    dt = 1 / 252
    num_steps = int(T / dt)
    hitting_count = 0
    price_paths = []
    volatility_paths = []

    for _ in range(num_simulations):
        price_path, volatility_path = simulate_price_path(
            S0, mu, sigma, T, dt, num_steps, lambda_jumps, jump_mean,
            jump_std, kappa, theta, vol_of_vol, rho
        )
        price_paths.append(price_path)
        volatility_paths.append(volatility_path)

        if any(price >= target_price for price in price_path):
            hitting_count += 1

    probability = hitting_count / num_simulations

    option_price = None
    if option_type and strike and risk_free_rate:
        mu = mu - risk_free_rate
        final_prices = []
    for path in price_paths: #iterating directly over the numpy array
        final_price = path[-1] #take the last price, regardless if it is a float or not
        if isinstance(final_price, np.ndarray): #if it is an array, make it a float
            final_prices.append(float(final_price[-1]))
        else: #if it is a float, keep it
            final_prices.append(float(final_price))

        payoffs = []
        for price in final_prices:
            if option_type == 'call':
                payoff = max(price - strike, 0)
            elif option_type == 'put':
                payoff = max(strike - price, 0)
            else:
                raise ValueError("Invalid option_type. Must be 'call' or 'put'.")
            payoffs.append(payoff)
        
        payoffs = np.array(payoffs) #Added conversion to numpy array
        option_price = np.mean(payoffs) * np.exp(-risk_free_rate * T) #using np.mean on numpy array
        strike_count = 0
        for price in payoffs:
            if price > 0:
                strike_count += 1
        option_prob = strike_count / num_simulations
    return probability, price_paths, volatility_paths, option_price, payoffs, option_prob

@st.cache_data
def load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)


st.title("Options Pricing and Monte Carlo Simulation App")
st.markdown("Welcome to the Options Pricing App! This app uses the Heston model and Monte Carlo simulation to simulate stock price paths and option prices.")

st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(np.datetime64("2023-12-26")).date())
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(np.datetime64("2024-12-26")).date())
target_price = st.sidebar.number_input("Target Price", value=605.0)
num_simulations = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)
num_days = st.sidebar.number_input("Days to Simulate", value=13, step=1)
lambda_jumps = st.sidebar.number_input("Lambda Jumps", value=1.0)
jump_mean = st.sidebar.number_input("Jump Mean", value=0.0)
jump_std = st.sidebar.number_input("Jump Std", value=0.1)

#Option Parameters
risk_free_rate = st.sidebar.number_input("Risk Free Rate", value=0.004, step = 0.001)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
strike_price = st.sidebar.number_input("Strike Price", value = 605.0)
market_implied_volatility = st.sidebar.number_input("Market Implied Volatility", value = 0.15, step=0.01)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Loading data and estimating parameters..."):
        data_file = load_data(ticker, start_date, end_date)
    
        # Estimate Heston parameters using Method of Moments
        kappa, theta, vol_of_vol, rho = estimate_heston_parameters_mom(data_file)
        st.caption(f"Estimated Heston parameters (MoM): kappa={kappa:.4f}, theta={theta:.4f}, vol_of_vol={vol_of_vol:.4f}, rho={rho:.4f}")

    with st.spinner("Running Monte Carlo Simulation..."):
        probability, price_paths, volatility_paths, option_price, payoffs, option_prob = monte_carlo_simulation(
            data_file, target_price, num_simulations, num_days, lambda_jumps, jump_mean,
            jump_std, kappa, theta, vol_of_vol, rho, option_type, strike_price, risk_free_rate
        )
    mu, sigma, S0 = estimate_drift_volatility(data_file)
    T = num_days / 252
   
    colp, cola, cola1, colb, colb1 = st.columns(5)
    with cola:
        st.metric("Calculated Drift:", f"{mu:.2f}", border =False)
    with colb:
        st.metric("Calculated Vol.:", f"{sigma:.2%}", border = False)
    
    colc, cold, cole = st.columns(3)
    with colc:
        st.metric("Target Price:", f"${target_price}", border = True)
    with cold:
        st.metric("Days:", f"{num_days}", border = True)
    with cole:
        st.metric("Probability of Reaching Target:", f"{probability:.2%}", border = True)
    if option_price:
        st.markdown("------------------------------------------------------------------------------")

        st.subheader(f":green[$**{strike_price}**] *{option_type}* with expiry in **{num_days}** days: ")

    # Calculate Black-Scholes Option price
    
    # --- Implied Volatility Calculations ---
    # Calculate implied volatility from the Heston model price
        model_implied_volatility = calculate_implied_volatility(S0, strike_price, T, risk_free_rate, option_price, option_type)

        if model_implied_volatility:
            black_scholes_price_model_iv = black_scholes_option_price(S0, strike_price, T, risk_free_rate, model_implied_volatility, option_type)
            black_scholes_price_model_iv = black_scholes_price_model_iv.item()
        else:
                st.write("Could not calculate implied volatility from the model, so Black-Scholes using model implied volatility is skipped")

            # --- Black-Scholes Price with Market Implied Volatility (if provided)---
        if market_implied_volatility is not None:
            black_scholes_price_market_iv = black_scholes_option_price(S0, strike_price, T, risk_free_rate, market_implied_volatility, option_type)
            black_scholes_price_market_iv = black_scholes_price_market_iv.item()

        else:
            st.write("Market implied volatility not provided, so Black-Scholes with market implied vol is skipped.")
        col1, col2, col3 = st.columns(3)
        with col1:  
            st.metric(label="Simulated Worth:",value=f"{option_price:.4f}", delta=f"{option_prob:.2%} Probability", border = True)
        
        with col2:
            st.metric("Black-Scholes(Model Vol.) :", f"{black_scholes_price_model_iv:.4f}", border = True )
    
        with col3:
            st.metric("Black-Scholes(Market Vol.) :", f"{black_scholes_price_market_iv:.4f}", border = True)
       
    st.markdown("------------------------------------------------------------------------------")
    # --- Remaining code for plotting ---
    price_paths = np.array(price_paths) #converted to numpy array
    price_path_average = np.mean(price_paths, axis=0)
    lower_percentile = np.percentile(price_paths, 2.5, axis=0)
    lower_percentile = np.squeeze(lower_percentile)
    upper_percentile = np.percentile(price_paths, 97.5, axis=0)
    upper_percentile = np.squeeze(upper_percentile)

    final_prices = []
    for path in price_paths: #iterating directly over the numpy array
        final_price = path[-1]
        final_prices.append(final_price)
    
    final_prices = np.array(final_prices) #converted to numpy array
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    
    #Plotting Price Paths
    st.subheader("Simulated Price Paths")
    fig_paths, ax_paths = plt.subplots()
    for i in range(min(100, len(price_paths))):
        ax_paths.plot(price_paths[i])
    ax_paths.axhline(y=target_price, color='r', linestyle='--')
    ax_paths.set_xlabel("Days")
    ax_paths.set_ylabel("Price")
    ax_paths.set_title(f"Simulated Price Paths (Heston Model)")
    st.pyplot(fig_paths)

    #Plotting the average price path and confidence intervals
    st.subheader("Average Price Path")
    fig_average, ax_average = plt.subplots()
    ax_average.plot(price_path_average)
    ax_average.set_xlabel("Days")
    ax_average.set_ylabel("Price")
    ax_average.set_title(f"Average Price Path (Heston Model)")
    ax_average.axhline(y=target_price, color='r', linestyle='--')
    ax_average.fill_between(
        range(len(price_path_average)),
        lower_percentile,
        upper_percentile,
        color='skyblue',
        alpha=0.4,
        label='95% Confidence Interval'
    )
    st.pyplot(fig_average)

    #Plotting the final prices
    st.subheader("Final Price Distribution")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(final_prices, bins='auto', range=(lower_percentile[-1], upper_percentile[-1]))
    ax_hist.axvline(mean_final_price, color='r', linestyle='dashed', linewidth=2, label='Mean')
    ax_hist.axvline(median_final_price, color='g', linestyle='dashed', linewidth=2, label='Median')
    ax_hist.axvline(target_price, color='r', linestyle='solid', linewidth=2, label='Target')
    ax_hist.set_xlabel("Final Prices")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Final Price at Target Date")
    ax_hist.legend()
    st.pyplot(fig_hist)

if __name__ == '__main__':
    pass
