import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.signal import peak_widths

def baseline_als_optimized(y, lam, p, niter=10):
    """Function which uses curve fitting to determine the baseline of a
    dataset.

    Input: y axis values (array), lambda* value, rho** value

    *lambda values between 10^2-10^9 typical, related to sampling rate
    **rho values between 0.01 and 0.1 typical

    Output: baseline as a z axis
    -------------------------------------------------------------------------"""
    # Defining parameters and matrix for smoothing
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def peak_detector(file, SNR, wval, gain_adj, zoom = False):
    """Function which uses scipy peak detector to find peaks within a
    set of NP collision data.

    Input: .txt file (typically converted from .atf file format)
    NO HEADER ONLY DATA [That means delete all information except
    for the current and time values], desired signal to noise ratio of peaks,
    width value of peak (wval; tuneable), gain_adjustment value.
    If zoom = True:
    the plotted graph will be zoomed to the 300 ms interval with
    the highest density of detected peaks.

    Output: number of peaks, frequency, and average half-width are
    read out to screen; array of peak indices is output;
    optional: plot zoomed to show interval of highest peak density
    -------------------------------------------------------------------------"""
    # Loading file into a dataframe
    df = pd.read_csv(file, sep="\t", header=None)
    # Gain adjustment
    df['gain_adj_y'] = df[1] * gain_adj
    # Finding Baseline
    bline = baseline_als_optimized(df['gain_adj_y'], 10e9, 0.001)
    # Setting height threshold based on desired S/N Ratio
    hthresh = bline + SNR * df['gain_adj_y'].std()
    # Using scipy.signal.find_peaks() to analyze current data
    peaks = find_peaks(df['gain_adj_y'], height=hthresh, width=wval)
    # obtaining/adjusting coordinate axes (dt = 0.01 ms)
    min_x_ind = np.min(df[0])/0.00001
    xcoords = (peaks[0] + min_x_ind) * 0.00001
    ycoords = peaks[1]['peak_heights']
    peak_indices = peaks[0]
    # Finding average peak height
    av_ph = np.mean(ycoords)
    std_ph = np.std(ycoords)
    # Calculate number of peaks
    npeaks = len(xcoords)
    freq = npeaks/(np.max(df[0]) - np.min(df[0]))

    # Find period of time with most peaks using histogram function
    # Determining number of bins (based on 300 ms window)
    nbins = int(np.max(xcoords)/0.3)
    # compute histogram
    pop, left_edges = np.histogram(xcoords, nbins)
    # Find zoom window
    max_pop = np.max(pop)
    max_ind = pop.argmax()
    right_edge = max_ind + 1
    xmin = left_edges[max_ind]
    xmax = left_edges[right_edge]

    # Plotting original trace
    plt.plot(df[0],df['gain_adj_y'])
    # Adding points to graph
    plt.scatter(xcoords,ycoords, c = 'red')
    # Labeling axes
    plt.ylabel('Current (pA)')
    plt.xlabel('Time(s)')
    if zoom is True:
        plt.xlim(xmin, xmax)
    else:
        pass

    print("\nTotal number of detected peaks: " + str(npeaks))
    print('\nFrequency of peaks (Hz): ' + str(freq))
    print("\nAverage peak height (pA): " + str(av_ph) + ' SD: ' + str(std_ph))
    return [npeaks, freq, peak_indices, df['gain_adj_y'], bline]


def find_half_widths(file, SNR, wval, gain_adj):
    '''Function which uses scipy.signal.peak_widths to find the
    half-widths of peaks detected within an amperometry trace

    Input: .txt file (typically converted from .atf file format)
    NO HEADER ONLY DATA, desired signal to noise ratio of peaks

    Output: number of peaks, frequency, and average half-width
    are read out to screen; half-width array in milliseconds is output
    ---------------------------------------------------------------------'''
    # Using Peak Detector to get Peaks
    n, f, ind, amps, bline = peak_detector(file, SNR, wval, gain_adj)
    # Loading file into a dataframe
    df = pd.read_csv(file, sep="\t", header=None)
    # Use scipy.signal.peak_widths on data (half-height)
    hws, rel_h, left, right = peak_widths(df[1], ind, rel_height = 0.5)
    # Adjust halfwidths to ms time scale
    halfwidths = hws * 0.01
    av_hw = np.mean(halfwidths)
    std_hw = np.std(halfwidths)
    print("\nAverage Half-width (ms): " + str(av_hw) + ' SD: ' + str(std_hw))
    return halfwidths


def calc_peak_area(file, SNR, wval, gain_adj):
    '''Function which uses scipy.signal.peak_widths to find the bounds of peaks
    detected within an amperometry trace and consequently calculates peak area
    using Simpson Integration (trapezoidal approximation)

    Input: .txt file (typically converted from .atf file format)
    NO HEADER ONLY DATA, desired signal to noise ratio of peaks

    Output: list of peak areas for each peak in the dataset
    ----------------------------------------------------------------'''
    # Creating Array output for peak areas
    peak_areas = []
    # Using Peak Detector to get Peaks
    n, f, ind, amps, bline = peak_detector(file, SNR, wval, gain_adj)
    # Loading file into a dataframe
    df = pd.read_csv(file, sep="\t", header=None)
    # Gain adjustment
    df['gain_adj_y'] = df[1] * gain_adj
    # Defining baseline as a horizontal line at y = mean of data set (will need to update)
    b = df['gain_adj_y'].mean()
    # Use scipy.signal.peak_widths on data (full height)
    hws, rel_h, left, right = peak_widths(df[1], ind, rel_height = 0.95)
    # Round left/right indices to callable integers
    left = left.astype(int)
    right = right.astype(int)
    # Iterating over each peak to calculate peak area
    for i in range(len(ind)):
        # Setting arrays needed for Simpson integration function
        y0 = df['gain_adj_y'].iloc[left[i]:right[i]]
        x0 = df[0].iloc[left[i]:right[i]]
        # Setting bounds for baseline integration
        xi = df[0][left[i]]
        xf = df[0][right[i]]
        # Integrate current trace
        T = scipy.integrate.simps(y0, x0)
        # Integrate baseline (assuming baseline is constant)
        B = b * (xf - xi)
        # Adding peak area to new array
        area = (T - B) * 1000
        peak_areas.append(area)
    # Replacing negative values with NaN
    peak_areas = np.asarray(peak_areas)
    peak_areas = np.where(peak_areas < 0, np.nan, peak_areas)
    # Computing average value
    av_pa = np.nanmean(peak_areas)
    std_pa = np.nanstd(peak_areas)
    print("\nAverage peak area (fC): " + str(av_pa)
         + ' SD: ' + str(std_pa))
    return peak_areas


def calc_particle_rads(array, unit, metal):
    """Takes an array of calculated charges from an amperometric
    trace and estimates the radius based on the atomic mass and density
    of the given metal.

    Inputs: array of charges,
    [THE FOLLOWING MUST BE STRINGS]
    unit of charge (prefix lowercase, suffix uppercase;
    i.e. fC [femtocoulomb], pC [picocoulomb]),
    type of metal (element symbol i.e. Ag (silver))

     Outputs: array of particle sizes in nm
     ----------------------------------------------------------------------"""
    # Organizing tuneable values into a dictionary
    # Adjusting for unit prefix so that final result is in centimeters
    unit_dict = {"fC" : 10**-15,
                "pC": 10**-12,
                "nC": 10**-9}
    # Atomic mass in g/mol
    atom_mass_dict = {"Ag" : 107.868,
                     "Au" : 196.966,
                     "Pt" : 194.965}
    # Density in g/cm^3
    density_dict = {"Ag" : 10.49,
                   "Au" : 19.32,
                   "Pt" : 21.45}
    # Defining Faraday's Constant (C/mol)
    F = 96500
    # Defining values for calculation
    am = atom_mass_dict[metal]
    dens = density_dict[metal]
    # Adjusting input unit to coulombs
    charges = array * unit_dict[unit]
    # Calculating radii
    num = charges * 3 * am
    denom = 4 * 3.1415 * dens * F
    rads = (num/denom)**(1/3) / 10**-7
    av_rad = np.nanmean(rads)
    std_rads = np.nanstd(rads)
    print("Average estimated particle radius (nm): " + str(av_rad)
         + ' SD: ' + str(std_rads))
    return rads
