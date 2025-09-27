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
# ## Get precalculated 3-layer EBM parameters

# %%
ebm3_df = pd.read_csv('../data/4xCO2_cummins_ebm3_cmip6.csv')

# %% [markdown]
# ## Try iterative example with CanESM5
#
# First things to note - the model does not seem to be in equilibrium in 1850, and of course there is the fact that temperatures are not given absolute
#
# We subtract the mean of 1850-1900 to make anomaly time series. Would be interesting to see if Hege's drift-corrected anomalies relative to the piControl run give different results.

# %%
hadgem3_model_tas = cmip6_df.loc[
    cmip6_df["model_p_f"] == "HadGEM3-GC31-LL_p1_f3", "tas_ensmean_p_f"
].values

# %%
hadgem3_model_net = cmip6_df.loc[
    cmip6_df["model_p_f"] == "HadGEM3-GC31-LL_p1_f3", "net_ensmean_p_f"
].values

# %%
pl.plot(hadgem3_model_tas)

# %%
pl.plot(hadgem3_model_net)

# %%
model = 'HadGEM3-GC31-LL'
run = 'r1i1p1f3'

# %%
row = ebm3_df.loc[(ebm3_df["model"]==model) & (ebm3_df["run"]==run)]

C1 = row.C1.values[0]
C2 = row.C2.values[0]
C3 = row.C3.values[0]
kappa1 = row.kappa1.values[0]
kappa2 = row.kappa2.values[0]
kappa3 = row.kappa3.values[0]
epsilon = row.epsilon.values

# these ones define autocorrelation etc, should not be important in deterministic EBM
gamma = row.gamma.values
sigma_xi = row.sigma_xi.values
sigma_eta = row.sigma_eta.values

# this one is not used except to estimate ECS/TCR
F_4xCO2 = row.F_4xCO2.values

# %%
ebm3 = EnergyBalanceModel(
    ocean_heat_capacity=[C1, C2, C3],
    ocean_heat_transfer=[kappa1, kappa2, kappa3],
    deep_ocean_efficacy=epsilon,
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# %%
ebm3.emergent_parameters()

# %%
ebm3.ecs

# %%
ebm3.tcr

# %%
hadgem3_model_net_anomaly = hadgem3_model_net - hadgem3_model_net[:51].mean()

# %%
hadgem3_model_tas_anomaly = hadgem3_model_tas - hadgem3_model_tas[:51].mean()

# %%
pl.plot(hadgem3_model_net_anomaly)

# %%
pl.plot(hadgem3_model_tas_anomaly)

# %% [markdown]
# ## Estimate of time-varying forcing
#
# Cummins, Geoffroy etc. show that
#
# $$N(t) = F(t) - \kappa_1 T_1(t) + (1-\epsilon) \kappa_k (T_{k-1}(t) - T_k(t))$$
#
# for the three layer model 
#
# $$N(t) = F(t) - \kappa_1 T_1(t) + (1-\epsilon) \kappa_3 (T_2(t) - T_3(t))$$
#
# $N(t)$, $T_1(t)$ are given by the climate model. $\epsilon$, $\kappa_1$, $\kappa_2$, $\kappa_3$ are given from the energy balance model fit to abrupt-4xCO2.
#
# We can drive the energy balance model with an estimate of time-varying forcing $\tilde{F}(t)$ to obtain responses $\tilde{T_1}(t)$, $\tilde{T_2}(t)$, $\tilde{T_3}(t)$ and $\tilde{N}(t)$. Then we want to jointly minimise the distance between the model and EBM estimates of TOA imbalance and surface temperature, i.e.
#
# Minimize $|\tilde{T_1}(t) - T(t)|$, $|\tilde{N}(t) - N(t)|$
#
# Proposed solution is to iterate to solve for $\tilde{F}(t)$ (taking $\tilde{T_2}(t)$ and $\tilde{T_3}(t)$ as latent).

# %% [markdown]
# ### Initial guess of forcing
#
# We rearrange the equation above to put in terms of $F(t)$. Before we run the EBM for the first time, we don't know what $T_2(t)$ and $T_3(t)$ are. For a first guess assume $\epsilon = 1$, giving $F(t) = N(t) + \kappa_1 T_1(t)$  (the standard Gregory regression)

# %%
F_init = hadgem3_model_net_anomaly + kappa1 * hadgem3_model_tas_anomaly

# %%
pl.plot(F_init)


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
ebm3 = EnergyBalanceModel(
    ocean_heat_capacity=[C1, C2, C3],
    ocean_heat_transfer=[kappa1, kappa2, kappa3],
    deep_ocean_efficacy=epsilon,
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# introduce our initial forcing guess and run it
ebm3.add_forcing(F_init, 1)
ebm3.run()

# %% [markdown]
# #### diagnostic plots

# %%
pl.plot(ebm3.toa_imbalance)
pl.plot(hadgem3_model_net_anomaly)

# %%
pl.plot(ebm3.temperature[:,0])
pl.plot(hadgem3_model_tas_anomaly)

# %%
pl.plot(ebm3.toa_imbalance - hadgem3_model_net_anomaly)

# %%
pl.plot(ebm3.temperature[:, 0] - hadgem3_model_tas_anomaly)

# %%
cost_function(ebm3.toa_imbalance, ebm3.temperature[:,0], hadgem3_model_net_anomaly, hadgem3_model_tas_anomaly)


# %% [markdown]
# ### Set up to optimize
#
# If we think about the whole ecosystem here, we iteratively supply the optimizer with our estimate of the ERF time series, the EBM calculates the time series of N and T1, we look at the value of the cost function, and we try to minimise this. So let's try and code it up.

# %%
def ebm_processor(forcing):
    ebm3 = EnergyBalanceModel(
        ocean_heat_capacity=[C1, C2, C3],
        ocean_heat_transfer=[kappa1, kappa2, kappa3],
        deep_ocean_efficacy=epsilon,
        gamma_autocorrelation=gamma,
        sigma_xi=sigma_xi,
        sigma_eta=sigma_eta,
        forcing_4co2=F_4xCO2,
        stochastic_run=False,
        seed=16
    )
    ebm3.add_forcing(forcing, 1)
    ebm3.run()

    return cost_function(ebm3.toa_imbalance, ebm3.temperature[:,0], hadgem3_model_net_anomaly, hadgem3_model_tas_anomaly)


# %%
opt_result = scipy.optimize.minimize(
    ebm_processor,
    F_init,
)

# %%
opt_result

# %%
pl.plot(opt_result.x)
pl.plot(F_init)

# %%
# define EBM
ebm3 = EnergyBalanceModel(
    ocean_heat_capacity=[C1, C2, C3],
    ocean_heat_transfer=[kappa1, kappa2, kappa3],
    deep_ocean_efficacy=epsilon,
    gamma_autocorrelation=gamma,
    sigma_xi=sigma_xi,
    sigma_eta=sigma_eta,
    forcing_4co2=F_4xCO2,
    stochastic_run=False,
    seed=16
)

# introduce our initial forcing guess and run it
ebm3.add_forcing(opt_result.x, 1)
ebm3.run()

# %%
pl.plot(ebm3.toa_imbalance)
pl.plot(hadgem3_model_net_anomaly)

# %%
pl.plot(ebm3.temperature[:,0])
pl.plot(hadgem3_model_tas_anomaly)

# %%
pl.plot(ebm3.toa_imbalance - hadgem3_model_net_anomaly)

# %%
pl.plot(ebm3.temperature[:,0] - hadgem3_model_tas_anomaly)

# %%
pl.plot(ebm3.temperature[:,0])
pl.plot(ebm3.temperature[:,1])
pl.plot(ebm3.temperature[:,2])

# %%
