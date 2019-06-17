def glm_mcmc_inference(df, iterations=5000):
  """
  Calculates the Markov Chain Monte Carlo trace of
  a Generalised Linear Model Bayesian linear regression
  model on supplied data.
  df: DataFrame containing the data
  iterations: Number of iterations to carry out MCMC for
  """
  # Use PyMC3 to construct a model context
  basic_model = pm.Model()
  with basic_model:
    # Create the glm using the Patsy model syntax
    # We use a Normal distribution for the likelihood
    pm.glm.glm("y ~ x", df, family=pm.glm.families.Normal())
    # Use Maximum A Posteriori (MAP) optimisation
    # as initial value for MCMC
    start = pm.find_MAP()

    # Use the No-U-Turn Sampler
    step = pm.NUTS()
    # Calculate the trace
    trace = pm.sample(
    iterations, step, start,
    random_seed=42, progressbar=True
    )

  return trace

if __name__ == "__main__":
    # These are our "true" parameters
    beta_0 = 1.0 # Intercept
    beta_1 = 2.0 # Slope
    # Simulate 100 data points, with a variance of 0.5
    N = 100
    eps_sigma_sq = 0.5
    # Simulate the "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)
    # Plot the data, and a frequentist linear regression fit
    # using the seaborn package
    sns.lmplot(x="x", y="y", data=df, size=10)
    plt.xlim(0.0, 1.0)