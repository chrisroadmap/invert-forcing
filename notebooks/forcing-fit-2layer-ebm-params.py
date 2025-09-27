# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit energy balance model to estimate forcing

# %% [markdown]
# ## Get imports

# %%
import pickle

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize

from fair.energy_balance_model import EnergyBalanceModel

# %% [markdown]
# ## Get pre-computed data from Gergana

# %%
with open('../data/tas_net_CMIP6.pkl', 'rb') as fp:
    cmip6_df = pickle.load(fp)

# %%
cmip6_df

# %%
models = cmip6_df["model_p_f"].unique()
models

# %% [markdown]
# ## Get precalculated 2-layer EBM parameters

# %%
ebm2_df = pd.read_csv('../data/4xCO2_cummins_ebm2_cmip6.csv')

# %% [markdown]
# ## Try iterative example with CanESM5
#
# First things to note - the model does not seem to be in equilibrium in 1850, and of course there is the fact that temperatures are not given absolute
#
# We subtract the mean of 1850-1900 to make anomaly time series. Would be interesting to see if Hege's drift-corrected anomalies relative to the piControl run give different results.

# %%
canesm5_model_tas = cmip6_df.loc[
    cmip6_df["model_p_f"] == "CanESM5_p2_f1", "tas_ensmean_p_f"
].values

# %%
canesm5_model_net = cmip6_df.loc[
    cmip6_df["model_p_f"] == "CanESM5_p2_f1", "net_ensmean_p_f"
].values

# %%
pl.plot(canesm5_model_tas)

# %%
pl.plot(canesm5_model_net)

# %%
model = 'CanESM5'
run = 'r1i1p2f1'

# %%
row = ebm2_df.loc[(ebm2_df["model"]==model) & (ebm2_df["run"]==run)]

C1 = row.C1.values[0]
C2 = row.C2.values[0]
kappa1 = row.kappa1.values[0]
kappa2 = row.kappa2.values[0]
epsilon = row.epsilon.values[0]

# these ones define autocorrelation etc, should not be important in deterministic EBM
gamma = row.gamma.values[0]
sigma_xi = row.sigma_xi.values[0]
sigma_eta = row.sigma_eta.values[0]

# this one is not used except to estimate ECS/TCR
F_4xCO2 = row.F_4xCO2.values[0]

# %%
ebm2 = EnergyBalanceModel(
    ocean_heat_capacity=[C1, C2],
    ocean_heat_transfer=[kappa1, kappa2],
    deep_ocean_efficacy=epsilon,
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# %%
ebm2.emergent_parameters()

# %%
ebm2.ecs

# %%
ebm2.tcr

# %%
canesm5_model_net_anomaly = canesm5_model_net - canesm5_model_net[:51].mean()

# %%
canesm5_model_tas_anomaly = canesm5_model_tas - canesm5_model_tas[:51].mean()

# %%
pl.plot(canesm5_model_net_anomaly)

# %%
pl.plot(canesm5_model_tas_anomaly)

# %% [markdown]
# ## Estimate of time-varying forcing
#
# Cummins, Geoffroy etc. show that
#
# $$N(t) = F(t) - \kappa_1 T_1(t) + (1-\epsilon) \kappa_k (T_{k-1}(t) - T_k(t))$$
#
# for the two layer model 
#
# $$N(t) = F(t) - \kappa_1 T_1(t) + (1-\epsilon) \kappa_2 (T_1(t) - T_2(t))$$
#
# $N(t)$, $T_1(t)$ are given by the climate model. $\epsilon$, $\kappa_1$ and $\kappa_2$ are given from the energy balance model fit to abrupt-4xCO2.
#
# We can drive the energy balance model with an estimate of time-varying forcing $\tilde{F}(t)$ to obtain responses $\tilde{T_1}(t)$, $\tilde{T_2}(t)$ and $\tilde{N}(t)$. Then we want to jointly minimise the distance between the model and EBM estimates of TOA imbalance and surface temperature, i.e.
#
# Minimize $|\tilde{T_1}(t) - T(t)|$, $|\tilde{N}(t) - N(t)|$
#
# Proposed solution is to iterate to solve for $\tilde{F}(t)$ (taking $\tilde{T_2}(t)$ as latent).

# %% [markdown]
# ### Initial guess of forcing
#
# We rearrange the equation above to put in terms of $F(t)$. Before we run the EBM for the first time, we don't know what $T_2(t)$ is. For a first guess we have two options:
#
# 1. assume $T_2(t) = 0$, giving $F(t) = N(t) + (\kappa_1 + (\epsilon - 1) \kappa_2) T_1(t)$ (the Gregory and Mitchell transient relationship with an infinite deep ocean heat sink)
# 2. assume $\epsilon = 1$, giving $F(t) = N(t) + \kappa_1 T_1(t)$  (the standard Gregory regression)
#
# Let's try both.

# %%
F_init_case1 = canesm5_model_net_anomaly + (kappa1 + (epsilon - 1) * kappa2) * canesm5_model_tas_anomaly
F_init_case2 = canesm5_model_net_anomaly + kappa1 * canesm5_model_tas_anomaly

# %%
pl.plot(F_init_case1)
pl.plot(F_init_case2)


# %% [markdown]
# ### Define our cost function to minimise
#
# Is jointly minimising RMSE the way to go?
#
# $$
# \text{Joint RMSE} = \sqrt{\frac{\sum_{i=1}^{N} (f_{1i} - a_{1i})^2 + \sum_{j=1}^{M} (f_{2j} - a_{2j})^2}{N+M}}
# $$
#
# \(f_{1i}\) and \(a_{1i}\) are the \(i^{th}\) predicted and actual values for the first variable. \(f_{2j}\) and \(a_{2j}\) are the \(j^{th}\) predicted and actual values for the second variable. \(N\) and \(M\) are the number of data points for each variable, respectively.

# %%
def cost_function(N_est, T_est, N_truth, T_truth):
    residual_N = N_est - N_truth
    residual_T = T_est - T_truth
    residual_N_sq = residual_N**2
    residual_T_sq = residual_T**2
    n_N = len(N_est)
    n_T = len(T_est)
    # there is no need to divide by (n_N + n_T), we just do it for completeness.
    return np.sqrt((np.sum(residual_N_sq) + np.sum(residual_T_sq)) / (n_N + n_T))


# %% [markdown]
# ### run one iteration manually to see if it works

# %%
# define EBM
ebm2 = EnergyBalanceModel(
    ocean_heat_capacity=[C1, C2],
    ocean_heat_transfer=[kappa1, kappa2],
    deep_ocean_efficacy=epsilon,
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# introduce our initial forcing guess and run it
ebm2.add_forcing(F_init_case1, 1)
ebm2.run()

# %% [markdown]
# #### diagnostic plots

# %%
pl.plot(ebm2.toa_imbalance)
pl.plot(canesm5_model_net_anomaly)

# %%
pl.plot(ebm2.temperature[:,0])
pl.plot(canesm5_model_tas_anomaly)

# %%
pl.plot(ebm2.toa_imbalance - canesm5_model_net_anomaly)

# %%
pl.plot(ebm2.temperature[:, 0] - canesm5_model_tas_anomaly)

# %%
cost_function(ebm2.toa_imbalance, ebm2.temperature[:,0], canesm5_model_net_anomaly, canesm5_model_tas_anomaly)


# %% [markdown]
# ### Set up to optimize
#
# If we think about the whole ecosystem here, we iteratively supply the optimizer with our estimate of the ERF time series, the EBM calculates the time series of N and T1, we look at the value of the cost function, and we try to minimise this. So let's try and code it up.

# %%
def ebm_processor(forcing_and_params):
    C1, C2, kappa1, kappa2, epsilon = forcing_and_params[:5]
    forcing = forcing_and_params[5:]
    ebm2 = EnergyBalanceModel(
        ocean_heat_capacity=[C1, C2],
        ocean_heat_transfer=[kappa1, kappa2],
        deep_ocean_efficacy=epsilon,
        gamma_autocorrelation=gamma,
        sigma_xi=sigma_xi,
        sigma_eta=sigma_eta,
        forcing_4co2=F_4xCO2,
        stochastic_run=False,
        seed=16
    )
    ebm2.add_forcing(forcing, 1)
    ebm2.run()

    return cost_function(ebm2.toa_imbalance, ebm2.temperature[:,0], canesm5_model_net_anomaly, canesm5_model_tas_anomaly)


# %%
#np.concatenate((np.array([C1, C2, kappa1, kappa2, epsilon]), F_init_case2))

# %%
forcing_and_params_init = np.concatenate((np.array([C1, C2, kappa1, kappa2, epsilon]), F_init_case2))
opt_result = scipy.optimize.minimize(
    ebm_processor,
    forcing_and_params_init,
)

# %%
opt_result

# %%
opt_result.x[0], opt_result.x[1], opt_result.x[2], opt_result.x[3], opt_result.x[4] 

# %%
pl.plot(opt_result.x[5:])
pl.plot(F_init_case2)

# %%
# define EBM
ebm2 = EnergyBalanceModel(
    ocean_heat_capacity=[opt_result.x[0], opt_result.x[1]],
    ocean_heat_transfer=[opt_result.x[2], opt_result.x[3]],
    deep_ocean_efficacy=opt_result.x[4],
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# introduce our initial forcing guess and run it
ebm2.add_forcing(opt_result.x[5:], 1)
ebm2.run()

# %%
pl.plot(ebm2.toa_imbalance)
pl.plot(canesm5_model_net_anomaly)

# %%
pl.plot(ebm2.temperature[:,0])
pl.plot(canesm5_model_tas_anomaly)

# %%
pl.plot(ebm2.toa_imbalance - canesm5_model_net_anomaly)

# %%
pl.plot(ebm2.temperature[:,0] - canesm5_model_tas_anomaly)

# %%
