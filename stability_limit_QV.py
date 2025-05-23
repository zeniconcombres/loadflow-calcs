"""This script calculates the stability limit of a power system using the QV curve method.
It uses the SciPy solver to perform the calculations.

Author: Inez Zheng
Date: 2025-05-20
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.optimize as opt
from scipy.optimize import fsolve
import sys, os
from power_transfer import *

# Constants
SCR = 1.1
X_R = 5.49
# V = 1.05 # Receiving end voltage
Vth = 1.2328 # Sending end voltage - base case

TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
PROJECT = "BDWF1"
POWER_FLOW = -1 # default -1 is charging i.e. flow from Vth to V, 1 is discharging i.e. flow from V to Vth

# Basic calculations
# the values are in pu
Zth_abs = 1/SCR
Rth = Zth_abs / np.sqrt(1 + X_R**2)
Xth = Rth * X_R
# print(Rth, Xth)

alpha = Rth / Zth_abs**2
beta = Xth / Zth_abs**2

P = 1.0 # Active power (pu)

def equation(Q):
    lhs = Vth 
    rhs = np.sqrt(
        V**2 - 2*(P*Rth + Q*Xth) + (P**2 + Q**2)*(Rth**2 + Xth**2)/(V**2)
    )
    return lhs - rhs

V_array = np.arange(0.8, 1.4, 0.01)
V_df = pd.DataFrame(V_array, columns=["V"]).set_index("V")
for V in V_array:
    # Calculate the dQdV curve
    dQdV = power_transfer_partial_deriv_dQdV(P, V, Vth, alpha, beta)
    V_df.loc[V, "dQdV"] = dQdV
    # print(f"V: {V:.2f}, dQ/dV: {dQdV:.2f}")
    Q = (
        V*(V-Vth) - P*complex(Rth,Xth)
    ) / complex(Xth,-Rth)
    delta = np.arcsin(P*Xth/V)
    # Q = 1/Xth * (V*np.cos(delta)-1) if delta else None
    V_df.loc[V, "Q"] = Q
    print(f"V: {V:.2f}, dQ/dV: {dQdV:.2f}, Q: {abs(Q):.2f}, delta: {delta:.2f}")

# we actually need a new equation - rearranging the voltage drop 
# Q = power_transfer(theta, V, Vth, alpha, beta)


# Plot the QV curve
plot = True
if plot:
    plt.figure(figsize=(10, 6))
    # plt.plot(V_df.index, V_df["dQdV"], marker='x')
    plt.plot(V_df.index, V_df["Q"], marker='x')
    plt.title("QV Curve")
    plt.xlabel("Voltage (pu)")
    plt.ylabel("dQ/dV | Q (pu)")
    plt.grid()
    plt.axhline(0, color='red', linestyle='--', label='dQ/dV = 0')
    plt.show()
