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
V = 1.1
Vth = 0.9

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
    """ Calculate the active (P) and reactive (Q) power transfer in a power system.
        V (float): Voltage magnitude at the receiving end.
        Vth (float): Voltage magnitude at the sending end (Thevenin equivalent voltage).
        alpha (float): Real part of the admittance.
        beta (float): Imaginary part of the admittance.
        theta (float): Phase angle difference between the sending and receiving end voltages (in radians).
    
    Returns:
        tuple: A tuple containing:
            - P (float): Active power transfer.
            - Q (float): Reactive power transfer.
    """
    P = alpha*(V**2 - V*Vth*np.cos(theta)) + beta*(V*Vth*np.sin(theta))
    Q = beta*(V**2 - V*Vth*np.cos(theta)) - alpha*(V*Vth*np.sin(theta))
    return P, Q

def power_transfer_partial_deriv(P, V, Vth, alpha, beta, theta):
    """ Calculate the partial derivatives of power transfer with respect to 
        the phase angle (theta) and voltage (V).
            P (float): Active power.
            V (float): Voltage magnitude.
            Vth (float): Thevenin equivalent voltage magnitude.
            alpha (float): Real part of the admittance.
            beta (float): Imaginary part of the admittance.
            theta (float): Phase angle difference between the voltages.
            tuple: A tuple containing:
                - dPdtheta (float): Partial derivative of active power with respect to theta.
                - dQdV (float): Partial derivative of reactive power with respect to voltage.
    
    Returns:
        tuple: A tuple containing:
            - dPdtheta (float): Partial derivative of active power with respect to theta.
            - dQdV (float): Partial derivative of reactive power with respect to voltage.
    """
    dPdtheta = V*Vth*(alpha*np.sin(theta) + beta*np.cos(theta))
    dQdV = 2*beta*V - (
        ((Vth**2)*V / (Zth_abs**2)) + 2*P*alpha*V - 2*(alpha**2)*(V**3)
    )/(np.sqrt(
        ((Vth**2)*(V**2) / (Zth_abs**2)) - ((P**2) - 2*P*alpha*(V**2) + (alpha**2)*(V**4))
    ))
    return dPdtheta, dQdV

def generating_results(V, Vth, alpha, beta, theta_range):
    """ Generates a DataFrame containing power transfer results for a range of theta values.
            V (float): Voltage magnitude at the sending end (pu).
            Vth (float): Voltage magnitude at the receiving end (pu).
            alpha (float): Line resistance (pu).
            beta (float): Line reactance (pu).
            theta_range (iterable): Range of theta values in degrees.
        Returns:
            pandas.DataFrame: DataFrame containing the following columns:
                - "P (pu)": Active power transfer (pu).
                - "Q (pu)": Reactive power transfer (pu).
                - "dPdtheta (pu)": Partial derivative of active power with respect to theta (pu).
                - "dQdV (pu)": Partial derivative of reactive power with respect to voltage (pu).
                - "Theta (deg)": Theta values in degrees.
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
    """ Plots the power transfer results against the angle theta.
        Parameters:
            results (DataFrame): A pandas DataFrame containing the power transfer results with columns 
                            "Theta (deg)", "P (pu)", "Q (pu)", "dPdtheta (pu)", and "dQdV (pu)".
            print_fig (bool): If True, saves the figure as a PNG file. Default is False.
        Returns: None
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
    ax.set_title(f'{PROJECT} Power Transfer vs. Theta: V{V} Vth{Vth} SCR{SCR} XR{X_R}')
    ax.legend()
    ax.set_xlim([-90, 120])
    ax.set_ylim([-2.0, 2.0])
    fig.tight_layout()
    plt.show()
    print(TODAY)
    if print_fig:
        fig.savefig(f"{TODAY}_{PROJECT}_power_transfer_V{V}_Vth{Vth}_SCR{SCR}_XR{X_R}.png")

def plot_power_transfer_plotly(results, print_fig=False):
    """ Plots power transfer characteristics using Plotly.
            results (dict): A dictionary containing the following keys:
                - "Theta (deg)": List or array of theta values in degrees.
                - "P (pu)": List or array of active power values in per unit.
                - "Q (pu)": List or array of reactive power values in per unit.
                - "dPdtheta (pu)": List or array of dP/dTheta values in per unit.
                - "dQdV (pu)": List or array of dQ/dV values in per unit.
            print_fig (bool, optional): If True, saves the plot as an HTML file. Defaults to False.
        Returns:
            None
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
        title=f'{PROJECT} Power Transfer vs. Theta: V{V} Vth{Vth} SCR{SCR} XR{X_R}',
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
        fig.write_html(f"{TODAY}_{PROJECT}_power_transfer_V{V}_Vth{Vth}_SCR{SCR}_XR{X_R}.html")

if __name__ == "__main__":
    print_fig = False
    # print(os.getcwd())
    # plot_power_transfer(generating_results(V, Vth, alpha, beta, range(-90, 120)))
    plot_power_transfer_plotly(generating_results(V, Vth, alpha, beta, range(-90, 120)),print_fig=print_fig)