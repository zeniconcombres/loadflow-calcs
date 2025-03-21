"""This script is designed for static load flow calculations of a two bus system.
The purpose and use case is to determine operating boundaries and angle limitations
for various SCR and X/R operating scenarios with respect to P and Q flows.

Author: Inez Zheng
Date Created: 2025-03-21"""

import os, sys
import pandas as pd, numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Constants
# SCR = 1.2
# X_R = 3
SCR = 1.1
X_R = 5.49
V = 1.0
Vth = 1.0

TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
PROJECT = "BDWF1"

# Basic calculations
# the values are in pu
Zth_abs = 1/SCR
Rth = Zth_abs / np.sqrt(1 + X_R**2)
Xth = Rth * X_R

alpha = Rth / Zth_abs**2
beta = Xth / Zth_abs**2

# Calculate the power transfer
def power_transfer(V, Vth, alpha, beta, theta):
    P = alpha*(V**2 - V*Vth*np.cos(theta)) + beta*(V*Vth*np.sin(theta))
    Q = beta*(V**2 - V*Vth*np.cos(theta)) - alpha*(V*Vth*np.sin(theta))
    return P, Q

def power_transfer_partial_deriv(P, V, Vth, alpha, beta, theta):
    dPdtheta = V*Vth*(alpha*np.sin(theta) + beta*np.cos(theta))
    dQdV = 2*beta*V - (
        ((Vth**2)*V / (Zth_abs**2)) + 2*P*alpha*V - 2*(alpha**2)*(V**3)
    )/(np.sqrt(
        ((Vth**2)*(V**2) / (Zth_abs**2)) - ((P**2) - 2*P*alpha*(V**2) + (alpha**2)*(V**4))
    ))
    return dPdtheta, dQdV

def generating_results(V, Vth, alpha, beta, theta_range):
    """_summary_

    Args:
        V (_type_): _description_
        Vth (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        theta_range (_type_): needs to be a range of values for theta in degrees as a default.
    """
    # Generating the dataframe
    results_dict = {"P (pu)":[], "Q (pu)":[],"dPdtheta (pu)":[], "dQdV (pu)":[]}
    results = df.from_dict(results_dict)
    results["Theta (deg)"] = theta_range

    # Calculating the power transfer values for each theta
    for theta in theta_range:
        P, Q = power_transfer(V, Vth, alpha, beta, np.radians(theta))
        results.loc[results["Theta (deg)"] == theta,"P (pu)"] = P
        results.loc[results["Theta (deg)"] == theta,"Q (pu)"] = Q
        # print(f"Theta: {theta} deg, P: {P} pu, Q: {Q} pu")

        dPdtheta, dQdV = power_transfer_partial_deriv(P, V, Vth, alpha, beta, np.radians(theta))
        results.loc[results["Theta (deg)"] == theta,"dPdtheta (pu)"] = dPdtheta
        results.loc[results["Theta (deg)"] == theta,"dQdV (pu)"] = dQdV
        # print(f"Theta: {theta} deg, dPdtheta: {dPdtheta} pu, dQdV: {dQdV} pu")
    return results

def plot_power_transfer(results, print_fig=False):
    """_summary_

    Args:
        results (_type_): _description_
    """
    # Plotting the results
    fig, ax = plt.subplots(1,1)
    results.plot(ax=ax, x="Theta (deg)", y=["P (pu)", "Q (pu)"], linestyle='--')
    results.plot(ax=ax, x="Theta (deg)", y=["dPdtheta (pu)", "dQdV (pu)"], linestyle='-')
    ax.axhline(y=1.0, color='magenta', linestyle=':', label='Pmax 1.0pu')
    ax.axhline(y=0.395, color='cyan', linestyle=':', label='Qmax 0.395pu')

    # Plot labels
    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('P | Q (pu)')
    ax.axvline(x=0, color='k', linestyle='-')
    ax.axhline(y=0, color='k', linestyle='-')
    ax.set_title(f'{PROJECT} Power Transfer vs. Theta')
    ax.legend()
    ax.set_xlim([-90, 120])
    ax.set_ylim([-2.0, 2.0])
    fig.tight_layout()
    plt.show()
    print(TODAY)
    if print_fig:
        fig.savefig(f"{TODAY}_{PROJECT}_power_transfer.png")

def plot_power_transfer_plotly(results, print_fig=False):
    """_summary_

    Args:
        results (_type_): _description_
    """
    fig = go.Figure()

    # Add traces for P (pu) and Q (pu)
    fig.add_trace(go.Scatter(x=results["Theta (deg)"], y=results["P (pu)"], mode='lines', name='P (pu)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=results["Theta (deg)"], y=results["Q (pu)"], mode='lines', name='Q (pu)', line=dict(dash='dash')))

    # Add traces for dPdtheta (pu) and dQdV (pu)
    fig.add_trace(go.Scatter(x=results["Theta (deg)"], y=results["dPdtheta (pu)"], mode='lines', name='dPdtheta (pu)'))
    fig.add_trace(go.Scatter(x=results["Theta (deg)"], y=results["dQdV (pu)"], mode='lines', name='dQdV (pu)'))

    # Add horizontal lines
    fig.add_hline(y=1.0, line=dict(color='magenta', dash='dot'), annotation_text='Pmax 1.0pu', annotation_position='bottom right')
    fig.add_hline(y=0.395, line=dict(color='cyan', dash='dot'), annotation_text='Qmax 0.395pu', annotation_position='bottom right')

    # Add vertical and horizontal lines at zero
    fig.add_vline(x=0, line=dict(color='black'))
    fig.add_hline(y=0, line=dict(color='black'))

    # Update layout
    fig.update_layout(
        title=f'{PROJECT} Power Transfer vs. Theta',
        xaxis_title='Theta (deg)',
        yaxis_title='P | Q (pu)',
        xaxis=dict(range=[-90, 120]),
        yaxis=dict(range=[-2.0, 2.0]),
        legend=dict(x=0.01, y=0.99)
    )

    # Show the plot
    fig.show()

    # Save the plot as a PNG file if print_fig is True
    if print_fig:
        fig.write_html(f"{TODAY}_{PROJECT}_power_transfer.html")

if __name__ == "__main__":
    print_fig = False
    # print(os.getcwd())
    # plot_power_transfer(generating_results(V, Vth, alpha, beta, range(-90, 120)))
    plot_power_transfer_plotly(generating_results(V, Vth, alpha, beta, range(-90, 120)),print_fig=print_fig)