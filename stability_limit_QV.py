"""This script calculates the stability limit of a power system using the QV curve method.
It uses the power system analysis toolbox (PSAT) to perform the calculations.

Author: Inez Zheng
Date: 2025-05-20
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from psat import psat
# from psat import plot
# from psat import data
# from psat import runpsat
# from psat import qv_curve
import sys, os
from power_transfer import *

# Constants
SCR = 1.2
X_R = 3
V = 1.0 # Receiving end voltage
Vth = 0.9 # Sending end voltage

TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
PROJECT = "BDWF1"
POWER_FLOW = -1 # default -1 is charging i.e. flow from Vth to V, 1 is discharging i.e. flow from V to Vth

# Basic calculations
# the values are in pu
Zth_abs = 1/SCR
Rth = Zth_abs / np.sqrt(1 + X_R**2)
Xth = Rth * X_R

alpha = Rth / Zth_abs**2
beta = Xth / Zth_abs**2

dQdV = power_transfer_partial_deriv_dQdV(P, V, Vth, alpha, beta)


