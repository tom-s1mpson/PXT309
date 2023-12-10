import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#Set font size and use LaTeX
plt.rc('font', size=13)
plt.rc('text', usetex=False)

# Read data

# Read the data from the CSV file
half_mm_cavity = pd.read_csv(r'C:\Users\C23030253\OneDrive - Cardiff University\Desktop\PIV0.5.csv', sep=",", header=0)
three_quarter_mm_cavity = pd.read_csv(r'C:\Users\C23030253\OneDrive - Cardiff University\Desktop\PIV0.75.csv', sep=",", header=0)
# Set column names


# Extract wavelength and power columns
current_05 = half_mm_cavity["Current (A)"]
power_05 = half_mm_cavity["Power (W)"]
voltage_05 = half_mm_cavity["Voltage (V)"]

current_075 = three_quarter_mm_cavity["Current (A)"]
power_075 = three_quarter_mm_cavity["Power (W)"]
voltage_075 = three_quarter_mm_cavity["Voltage (V)"]

from sklearn.linear_model import LinearRegression

# Read the data from the CSV file
half_mm_cavity = pd.read_csv(r'C:\Users\C23030253\OneDrive - Cardiff University\Desktop\PIV0.5.csv', sep=",", header=0)
three_quarter_mm_cavity = pd.read_csv(r'C:\Users\C23030253\OneDrive - Cardiff University\Desktop\PIV0.75.csv', sep=",", header=0)

# Extract wavelength and power columns
current_05 = half_mm_cavity["Current (A)"]
power_05 = half_mm_cavity["Power (W)"]

current_075 = three_quarter_mm_cavity["Current (A)"]
power_075 = three_quarter_mm_cavity["Power (W)"]

# Create a linear regression model
model_05 = LinearRegression()
model_075 = LinearRegression()

# Fit the model to the linear part of the data
linear_range_05 = (current_05 >= 0.57) & (current_05 <= 1)
linear_range_075 = (current_075 >= 0.78) & (current_075 <= 1)

model_05.fit(current_05[linear_range_05].values.reshape(-1, 1), power_05[linear_range_05])
model_075.fit(current_075[linear_range_075].values.reshape(-1, 1), power_075[linear_range_075])

# Predict values using the linear regression model
predicted_power_05 = model_05.predict(current_05.values.reshape(-1, 1))
predicted_power_075 = model_075.predict(current_075.values.reshape(-1, 1))

fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=800)

# Figure 1: Power vs Current
axs[0].plot(current_05, 1e6*power_05, label="0.5 mm")
axs[0].plot(current_075, 1e6*power_075, label="0.75 mm", linestyle="--")
axs[0].plot(current_05, 1e6*predicted_power_05, linestyle="-", color="black", zorder=-1)
axs[0].plot(current_075, 1e6*predicted_power_075, linestyle="-", color="black", zorder=-1)
axs[0].set_xlabel("Current (A)")
axs[0].set_ylabel("Power (μW)")
axs[0].set_xlim(-0.01, 1.01)
axs[0].set_ylim(0e-5, 25)
axs[0].text(0.05, 0.9, "(a)", transform=axs[0].transAxes, weight='bold', ha='center')  # Add the label  # Add the label


# Figure 2: Voltage vs Current
axs[1].plot(current_05, voltage_05, label="0.50 mm")
axs[1].plot(current_075, voltage_075, label="0.75 mm", linestyle="--")
axs[1].set_xlabel("Current (A)")
axs[1].set_ylabel("Voltage (V)")
axs[1].set_xlim(-0.01, 1.01)
axs[1].set_ylim(0e-5, 3.5)
axs[1].text(0.025, 0.9, "(b)", transform=plt.gca().transAxes, weight='bold') 
axs[1].legend(frameon=True, title="Cavity Length", loc='lower right')

# Adjust layout
plt.tight_layout()
plt.show()

# Read data
spectrum = pd.read_csv('C:/Users/C23030253/Downloads/spectrum.txt', sep="	", header=1)

# Set column names
spectrum.columns = ["Wavelength (nm)", "Power (mW)"]

# Extract wavelength and power columns
wavelengths = spectrum["Wavelength (nm)"]
powers = spectrum["Power (mW)"]

# Find the index of the maximum power value
peak_index = np.argmax(powers)

# Get the wavelength value at the peak
peak_wavelength = wavelengths.iloc[peak_index]

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 2 / stddev)**2)

# Initial guess for the parameters
initial_params = [powers.max() * 1e6, peak_wavelength, 0.002]  # Convert max power to nanowatts

# Fit the Gaussian function to the data
params, covariance = curve_fit(gaussian, wavelengths, powers * 1e6, p0=initial_params)  # Convert powers to nanowatts

# Generate y values using the fitted parameters
fit_powers = gaussian(wavelengths, *params)

# Load data for the second spectrum
spectrum2 = pd.read_csv(r'C:\Users\C23030253\OneDrive - Cardiff University\Desktop\spectrum_narrow.csv', sep=",", header=1)

# Set column names
spectrum2.columns = ["Wavelength (nm)", "Power (mW)"]

# Extract wavelength and power columns
wavelengths2 = spectrum2["Wavelength (nm)"]
powers2 = spectrum2["Power (mW)"]

# Find the index of the maximum power value for the second spectrum
peak_index2 = np.argmax(powers2)

# Get the wavelength value at the peak for the second spectrum
peak_wavelength2 = wavelengths2.iloc[peak_index2]

# Initial guess for the parameters for the second spectrum
initial_params2 = [powers2.max() * 1e6, peak_wavelength2, 0.002]  # Convert max power to nanowatts

# Fit the Gaussian function to the data for the second spectrum
params2, covariance2 = curve_fit(gaussian, wavelengths2, powers2 * 1e6, p0=initial_params2)  # Convert powers to nanowatts

# Generate y values using the fitted parameters for the second spectrum
#Continuing from where we left off:


fit_powers2 = gaussian(wavelengths2, *params2)

# Create subplots
plt.figure(figsize=(12, 4), dpi=400)

# Plot the first subplot
plt.subplot(1, 2, 1)
plt.plot(wavelengths, powers * 1e6, color="black", label="Data (nW)")  # Convert powers to nanowatts
plt.plot(wavelengths, fit_powers, color="red", label="Gaussian Fit (nW)")  # Convert powers to nanowatts
plt.xlabel("Wavelength, $\lambda$ (nm)")
plt.ylabel("Power, $P$ (nW)")
plt.xlim(620, 630)

# Add a grey line at the peak wavelength
plt.axvline(x=peak_wavelength, color='grey', label=f'$\lambda_{{max}}$ = {peak_wavelength:.2f} $\pm$ 0.005 nm')
plt.text(0.05, 0.9, "(a)", transform=plt.gca().transAxes, weight='bold')  # Add bold (a)

plt.legend(frameon=False)

# Plot the second subplot with different xlim and no y-axis label
plt.subplot(1, 2, 2)
plt.plot(wavelengths2, powers2 * 1e6, color="black", label="Data (nW)")  # Convert powers to nanowatts
plt.plot(wavelengths2, fit_powers2, color="red", label="Gaussian Fit (nW)")  # Convert powers to nanowatts
plt.xlabel("Wavelength, $\lambda$ (nm)")
plt.xlim(623, 624)  # Different xlim for the second subplot

# Add a grey line at the peak wavelength for the second spectrum
plt.axvline(x=peak_wavelength2, color='grey', label=f'$\lambda_{{max}}$ = {peak_wavelength2:.3f} $\pm$ 0.001 nm')
plt.text(0.90, 0.9, "(b)", transform=plt.gca().transAxes, weight='bold')  # Add bold (b)

plt.legend(frameon=False)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()