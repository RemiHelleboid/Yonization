import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def find_nearest_vector(array, value):
  idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
  return idx, array[idx]


def smooth(y, box_pts=11):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def electric_field_profile(x_position):
    x_pos = x_position
    low_field = 1.0e2
    electric_field_x = 0
    PosJunc = 1.36e-4
    if (x_pos <= 1e-4):
        electric_field_x = low_field
    elif (x_pos > 1e-4 and x_pos <= 1.36e-4):
        electric_field_x = 1_624_982.38 * x_pos * 1e4 - 1_614_982.38
    elif (x_pos > 1.36e-4 and x_pos <= 1.49e-4):
        electric_field_x = -4_656_341.3 * x_pos * 1e4 + 6_946_969.7
    else:
        electric_field_x = low_field
    factor_mult = 1.3
    return factor_mult * electric_field_x


def func_electric_field(x_position):
    f = np.vectorize(electric_field_profile)
    return f(x_position)


def convolution_function(x_pos, x_init, lambda_p=10e-9, C_p=2.0/np.sqrt(np.pi)):
    y = (C_p / lambda_p) * np.exp((x_pos-x_init)**2 / lambda_p**2)
    return y


def effective_field(x_pos, x_init, x_line, electric_field_line):
    index_x_init, _ = find_nearest_vector(x_line, x_init)
    index_x_pos, _  = find_nearest_vector(x_line, x_pos)
    effective_field = np.trapz(electric_field_line[index_x_init:index_x_pos:], x_line[index_x_init:index_x_pos:])
    return effective_field


x_large = np.linspace(-2.0e-4, 2.0e-4, 1000)
profile_raw = func_electric_field(x_large)
profile_smooth = smooth(profile_raw, 51)

def function_electric_field(x_position):
    return np.interp(x_position, x_large, profile_smooth)

def electric_field_from_file(filename, x_position):
    x, field = np.loadtxt(filename, unpack=True, skiprows=1, delimiter=',')
    return np.interp(x_position, x, field, left=0.0, right=0.0)


if __name__ == "__main__":
    x_line = x_large
    y_electric_field = profile_smooth
    plt.plot(x_line, y_electric_field, label="Electric Field profile for tests")
    # plt.yscale("log")
    plt.title("Electric Field Profile")
    plt.xlabel("Position X ($\mu m$)")
    plt.xlabel("Electric Field ($V \cdot cm^{-1}$)")
    plt.show()
