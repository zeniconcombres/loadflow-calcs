"""This script calculates the stability limit of a power system using the QV curve method.
It uses the SciPy solver to perform the calculations.

Author: Inez Zheng
Date: 2025-05-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import fsolve
import sys, os
from power_transfer import *
from plotly.subplots import make_subplots

TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
PROJECT = "BDWF1"
POWER_FLOW = -1 # default -1 is charging i.e. flow from Vth to V, 1 is discharging i.e. flow from V to Vth

def SCR_calc(SCR=SCR, X_R=X_R):
    """Calculate the SCR based on the given parameters."""
    Zth_abs = 1/SCR
    Rth = Zth_abs / np.sqrt(1 + X_R**2)
    Xth = Rth * X_R
    return Rth, Xth

def equation_PV(Vth):
    lhs = Vth
    rhs = np.sqrt(
        V**2 - 2*(P*Rth + Q*Xth) + (P**2 + Q**2)*(Rth**2 + Xth**2)/(V**2)
    )
    return lhs - rhs

# Q = 0.2 # Reactive power (pu) TODO: make this a parameter input
def PV_plot(Q=0.3, SCR=SCR, Vth=Vth):
    """Calculate and plot the PV curve."""
    P_array = np.arange(0.0, 1.01, 0.01)
    P_df = pd.DataFrame(P_array, columns=["P"]).set_index("P")
    for P in P_array:
        Vth = fsolve(equation_PV, 0)[0] # initial guess is 0
        delta = np.arcsin(P*Xth/V)
        # Q = 1/Xth * (V*np.cos(delta)-1) if delta else None
        P_df.loc[P, "Vs"] = Vth
        print(f"Vs: {Vth:.2f}, P: {abs(P):.2f}, delta: {delta:.2f}")

    # Plot the PV curve in plotly
    plot_dyn = True
    print_html = True
    if plot_dyn:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=P_df.index,
            y=P_df["Vs"],
            mode='lines+markers',
            name='Slack voltage (pu)',
            line=dict(color='blue', width=2),
        ))
        # fig.add_hline(y=0.0, line=dict(color='red', dash='dot'), annotation_text='dQdV = 0', annotation_position='bottom right')
        # fig.add_hline(y=0.0, line=dict(color='red'), annotation_text='Q = 0', annotation_position='bottom right')
        fig.update_layout(
            title=f'{PROJECT} PV Curve at SCR = {SCR:.2f}, Vpoc = {V:.2f}pu and Q = {Q:.2f}pu',
            xaxis_title='P at POC (pu)',
            yaxis_title='Vslack (pu)',
            # xaxis=dict(range=[0.0, 1.4]),
            # yaxis=dict(range=[min(round(P_df["Vs"].min(),2)*1.05,0.0), round(P_df["P"].max(),2)*1.05]),
            legend=dict(x=0.01, y=0.99)  
        )
        if print_html:
            fig.write_html(f"{TODAY}_{PROJECT}_PVcurve_Q{Q:.2f}_Vpoc{V:.2f}_SCR{SCR:.2f}.html")
        # fig.show(config={
        #     'toImageButtonOptions': {
        #         'filename': f"{TODAY}_{PROJECT}_QVcurve_Vs{Vth:.2f}_SCR{SCR:.2f}",
        #         'format': "png"  # You can also set this to "jpeg", "svg", etc.
        #         }
        # })

def equation_QV(Q, Vth, P, V, Rth, Xth):
    lhs = Vth
    rhs = np.sqrt(
        V**2 - 2*(P*Rth + Q*Xth) + (P**2 + Q**2)*(Rth**2 + Xth**2)/(V**2)
    )
    return lhs - rhs                                     

def QV_calc(SCR=SCR, X_R=X_R, Vth=Vth, P=1.0):
    """Calculate the QV curve."""
    # Calculate the SCR and Xth based on the given parameters
    Rth, Xth = SCR_calc(SCR, X_R)
    # Define the voltage dataframe
    V_array = np.arange(0.4, 1.6, 0.01)
    V_df = pd.DataFrame(V_array, columns=["V"]).set_index("V")
    for V in V_array:
        # Calculate the dQdV curve
        dQdV = power_transfer_partial_deriv_dQdV(P, V, Vth, alpha, beta)
        V_df.loc[V, "dQdV"] = dQdV
        # print(f"V: {V:.2f}, dQ/dV: {dQdV:.2f}")
        Q = fsolve(equation_QV, 0, args=(Vth, P, V, Rth, Xth))[0] # initial guess is 0
        delta = np.arcsin(P*Xth/V)
        # Q = 1/Xth * (V*np.cos(delta)-1) if delta else None
        V_df.loc[V, "Q"] = Q
        V_df.loc[V, "delta"] = delta*180/np.pi  # Convert radians to degrees
        # Print the results
        print(f"V: {V:.2f}, dQ/dV: {dQdV:.2f}, Q: {abs(Q):.2f}, delta: {delta:.2f}")
    return V_df

def QV_plot(V_df, fig=None, SCR=1.1, X_R=X_R, Vth=1.1):
    """Plot the QV curve."""
    # Process the dataframe to get rid of nans
    V_df = V_df.dropna(subset=["Q", "dQdV", "delta"])

    # Plotting the QV curve
    if fig is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=V_df.index,
        y=V_df["Q"],
        mode='lines',
        name=f'Reactive power (Q) at Vs = {Vth:.2f}',
        # line=dict(color='blue', width=2),
        ), secondary_y=False,
    )
    # Find the lowest point in the Q curve
    min_Q = V_df["Q"].min()
    min_Q_voltage = V_df["Q"].idxmin()  # Corresponding voltage (x-axis)

    # Add a marker for the lowest point
    fig.add_trace(go.Scatter(
        x=[min_Q_voltage],
        y=[min_Q],
        mode='markers+text',
        marker=dict(color='red', size=5, symbol='x'),
        showlegend=False
        # text=f"Min Q: {min_Q:.2f} at V={min_Q_voltage:.2f}",
        # textposition="top center",
        # name="Lowest Point"
    ))
    # fig.add_hline(y=0.0, line=dict(color='red', dash='dot'), annotation_text='dQdV = 0', annotation_position='bottom right')
    fig.add_hline(y=0.0, line=dict(color='black'), annotation_text='Q = 0 pu', annotation_position='bottom right')
    fig.add_vline(x=1.04, line=dict(color='black'), annotation_text='Vpoc = 1.04 pu', annotation_position='bottom right')
    fig.update_layout(
        title=f'{PROJECT} QV Curve at SCR = {SCR:.2f} and XR = {X_R:.2f} with various Vpoc and Vs',
        xaxis_title='Voltage at POC (pu)',
        yaxis_title='Q (pu)',
        xaxis=dict(range=[0.85, 1.2]),
        # yaxis=dict(range=[min(round(V_df["Q"].min(),2)*1.05,0.0), round(V_df["Q"].max(),2)*1.05]),
        legend=dict(x=0.01, y=0.01)  
    )
    # fig.show(config={
    #     'toImageButtonOptions': {
    #         'filename': f"{TODAY}_{PROJECT}_QVcurve_Vs{Vth:.2f}_SCR{SCR:.2f}",
    #         'format': "png"  # You can also set this to "jpeg", "svg", etc.
    #         }
    # })
    return fig

# # Plotting with matplotlib
# plot = False
# if plot:
#     plt.figure(figsize=(10, 6))
#     # plt.plot(V_df.index, V_df["dQdV"], marker='x')
#     plt.plot(V_df.index, V_df["Q"], marker='x')
#     plt.title("QV Curve")
#     plt.xlabel("Voltage (pu)")
#     plt.ylabel("dQ/dV | Q (pu)")
#     plt.grid()
#     plt.axhline(0, color='red', linestyle='--', label='dQ/dV = 0')
#     plt.show()

if __name__ == "__main__":
    # System parameters
    System_situ = {
        "SCRmax": [4.64, 5.49],
        "SysNormal_SCRmin": [2.53, 5.49],
        "N-1_SCRmin": [1.1, 5.49]
    }
    V = 1.04 # Sending end voltage
    # Vth = 1.2328 # Receiving end voltage - base case
    Vth = 1.1 # Receiving end voltage - base case

    # Basic calculations
    # the values are in pu
    Zth_abs = 1/SCR
    Rth = Zth_abs / np.sqrt(1 + X_R**2)
    Xth = Rth * X_R
    # print(Rth, Xth)

    alpha = Rth / Zth_abs**2
    beta = Xth / Zth_abs**2

    P = 1.0 # Active power (pu)
    Q = 0.3 # Reactive power (pu)

    for SCR, X_R in System_situ.values():
        print(f"SCR: {SCR}, X_R: {X_R}")
        fig = None
        Vs_array = np.arange(0.85, 1.3, 0.05)
        Vs_df = pd.DataFrame(Vs_array, columns=["Vs"]).set_index("Vs")
        for Vs in Vs_array:
            # Run the QV calculation
            results_df = QV_calc(SCR=SCR, X_R=X_R, Vth=Vs)
            fig = QV_plot(results_df, fig=fig, SCR=SCR, X_R=X_R, Vth=Vs)
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df["delta"],
            mode='lines+markers',
            name='POC and Slack phase angle difference (degrees)',
            line=dict(color='purple', width=2),
            ), secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Phase Angle Difference (degrees)",  # Title for the secondary y-axis
            secondary_y=True
        )
        fig.update_layout(
            yaxis=dict(range=[-0.5, 1.0])
        )
        # Add a shaded region (mask) between x=0.99 and x=1.08
        fig.add_shape(
            type="rect",
            x0=0.99,  # Start of the mask on the x-axis
            x1=1.08,  # End of the mask on the x-axis
            y0=-0.5,     # Start of the mask on the y-axis
            y1=1.0,   # End of the mask on the y-axis (adjust as needed)
            fillcolor="rgba(255, 255, 0, 0.2)",  # Gray color with 20% opacity
            line=dict(width=0),  # No border
            layer="below"  # Place the mask below the data
        )

        print_html = True
        if print_html:
            fig.write_html(f"{TODAY}_{PROJECT}_QVcurve_VsScan_SCR{SCR:.2f}_XR{X_R:.2f}.html")