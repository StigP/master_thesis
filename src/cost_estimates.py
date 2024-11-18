"""Code for Appendix C delay costs based on the Eurocontrol
and University of Westminster values"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Common delay times for both gate and taxi
x_delay = np.array([5, 15, 30])  # Delay times

# Data from https://ansperformance.eu/economics/cba/standard-inputs/chapters/cost_of_delay.html#tbl-delay-cost-total-tact

# Data points for B738 (Boeing 737-800) - Delay at gate and taxi
y_b738_gate = np.array([107, 641, 2305])  # Delay cost values for B738 at gate
y_b738_taxi = np.array([178, 856, 2745])  # Delay cost values for B738 at taxi

# Data points for A332 (Airbus A330-200) - Delay at gate and taxi
y_a332_gate = np.array([213, 1176, 4218])  # Delay cost values for A332 at gate
y_a332_taxi = np.array([404, 1722, 5310])  # Delay cost values for A332 at taxi

# Transform of x values to include the quadratic term, no intercept
X_delay = np.column_stack((x_delay**2, x_delay))

# Fit linear regression models:
model_b738_gate = LinearRegression(fit_intercept=False).fit(X_delay, y_b738_gate)
model_a332_gate = LinearRegression(fit_intercept=False).fit(X_delay, y_a332_gate)
model_b738_taxi = LinearRegression(fit_intercept=False).fit(X_delay, y_b738_taxi)
model_a332_taxi = LinearRegression(fit_intercept=False).fit(X_delay, y_a332_taxi)

# Coeff:
a_b738_gate, b_b738_gate = model_b738_gate.coef_
a_a332_gate, b_a332_gate = model_a332_gate.coef_
a_b738_taxi, b_b738_taxi = model_b738_taxi.coef_
a_a332_taxi, b_a332_taxi = model_a332_taxi.coef_

# Plotting the quadratic curves from delay 0 to 30:
x_fit = np.linspace(0, 30, 100)
y_fit_b738_gate = a_b738_gate * x_fit**2 + b_b738_gate * x_fit
y_fit_a332_gate = a_a332_gate * x_fit**2 + b_a332_gate * x_fit
y_fit_b738_taxi = a_b738_taxi * x_fit**2 + b_b738_taxi * x_fit
y_fit_a332_taxi = a_a332_taxi * x_fit**2 + b_a332_taxi * x_fit

# Plot the quadratic curves for delay at gate
plt.figure(figsize=(10, 5))
plt.scatter(x_delay, y_b738_gate, color='red', label='B738 Gate Data Points')
plt.plot(x_fit, y_fit_b738_gate, color='red', linestyle='--', 
         label=f'B738 Gate Fit: y = {a_b738_gate:.2f}t² + {b_b738_gate:.2f}t')
plt.scatter(x_delay, y_a332_gate, color='blue', label='A332 Gate Data Points')
plt.plot(x_fit, y_fit_a332_gate, color='blue', linestyle='--', 
         label=f'A332 Gate Fit: y = {a_a332_gate:.2f}t² + {b_a332_gate:.2f}t')
plt.xlabel('Delay Time (t) minutes')
plt.ylabel('Cost (EUR)')
plt.legend()
plt.title('Fitted delay cost for B738 and A332 - Delay at Gate')
plt.show()

# Plot the quadratic curves for delay at taxi
plt.figure(figsize=(10, 5))
plt.scatter(x_delay, y_b738_taxi, color='red', label='B738 Taxi Data Points')
plt.plot(x_fit, y_fit_b738_taxi, color='red', linestyle='--', 
         label=f'B738 Taxi Fit: y = {a_b738_taxi:.2f}t² + {b_b738_taxi:.2f}t')
plt.scatter(x_delay, y_a332_taxi, color='blue', label='A332 Taxi Data Points')
plt.plot(x_fit, y_fit_a332_taxi, color='blue', linestyle='--', 
         label=f'A332 Taxi Fit: y = {a_a332_taxi:.2f}t² + {b_a332_taxi:.2f}t')
plt.xlabel('Delay Time (t) minutes')
plt.ylabel('Cost (EUR)')
plt.legend()
plt.title('Fitted delay cost for B738 and A332 - Delay at Taxi')
plt.show()
