import pandas as pd
import numpy as np
from itertools import groupby
from scipy.signal import convolve, windows,resample_poly, savgol_filter
from fractions import Fraction

def detect_blinks_and_noises(df,check_cols,sampling_rate = 1000, min_duration_blink = 200):
    """
    Detect blinks based on the NaN values.
    Blink definition: successive NaN for 200ms

        Input
        ------
        df : dataframe
        sampling_rate : sampleing rate of the eye tracker, Default value is 1000Hz
        min_duration_blink : minimum duration to define a blink in ms. Default value is 200ms. Other samples with NaN are defined as noise. 

        Output
        -------
        blink_profile : dict[str, Any]
            Returns self, useful for method cascading.
    """
    # find index of rows containing Nan values. 
    min_blink_duration_in_sample = int(sampling_rate/1000 * min_duration_blink) # convert ms to samples. 
    has_nans = df[check_cols].isna().any(axis=1) # whether each row contains Nan or not. 
    values=[] 
    lengths=[]
    for key, group in groupby(has_nans):  # group the same value together. Check the number of consecutive NaN
        values.append(key)                # the value (True or False) of this group
        lengths.append(len(list(group)))             # the length of each group
    cumulative_sum_np = np.cumsum(lengths)
    start_idx = cumulative_sum_np - np.array(lengths)
    end_idx = cumulative_sum_np - 1
    nan_profile = pd.DataFrame({
        "values":values,
        "lengths":lengths,
        "cum_idx":cumulative_sum_np,
        "start_idx":start_idx,
        "end_idx":end_idx
    })
    condition_true = nan_profile["values"] == True
    condition_blink = nan_profile["lengths"] >= min_blink_duration_in_sample
    blink_profiles = nan_profile[ condition_true & condition_blink]
    noise_profiles = nan_profile[ condition_true & ~condition_blink]

    return blink_profiles,noise_profiles

def remove_blinks(df,blink_profile,sampling_rate=1000,pre_post=50):
    """
    Pre and post duration of blink: 50ms

        Input
        ------
        df
        

        Output
        -------
        blink_profile
            Returns self, useful for method cascading.
    """
    if not blink_profile.empty:
        num_rows = df.shape[0]
        pre_post_blink_samples = int(sampling_rate/1000 * pre_post) # convert ms to samples. 
        blink_start_idx = blink_profile["start_idx"].to_numpy()
        blink_end_idx = blink_profile["end_idx"].to_numpy()
        # pre and post blink need to be removed as well.
        blink_start_idx = blink_start_idx - pre_post_blink_samples
        blink_start_idx = np.maximum(blink_start_idx,0)
        blink_end_idx = blink_end_idx + pre_post_blink_samples
        blink_end_idx = np.minimum(blink_end_idx,num_rows-1)
        assert len(blink_start_idx) == len(blink_end_idx)
        # check whether start and end idx has overlap. If so, merge to one 
        intervals = sorted([[s,e] for s,e in zip(blink_start_idx,blink_end_idx)])
        merged_intervals = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_merged_start, last_merged_end = merged_intervals[-1]
            if current_start <= last_merged_end:
                merged_intervals[-1][1] = max(current_end,last_merged_end)
            else:
                merged_intervals.append([current_start,current_end])
        drop_idx = []
        for start, end in merged_intervals:
            drop_idx.extend(np.arange(start,end+1))
        df_no_blink = df.drop(drop_idx,axis=0).reset_index(drop=True)
        num_rows_with_nan = df_no_blink.isnull().any(axis=1).sum()
        #print(f"num rows contain NaN after removing the blink: {num_rows_with_nan}")
        df_final = df_no_blink.dropna(axis=0).reset_index(drop=True)
    else:
        #print("No blinks detected, returning original DataFrame.")
        return df
    return df_final

def na_replacement(df, type, check_cols):
    """
    Replaces missing (NA) and infinite values in gaze point and pupil size measurements
    within a DataFrame based on the specified replacement type.

    Args:
        df (pd.DataFrame): The input DataFrame containing gaze and pupil data.
        type (str): The method for replacing NA values.
                    Can be "zero", "interpolate", or "locf".
        check_cols (list): A list of column headers to check for NA values.

    Returns:
        pd.DataFrame: The DataFrame with missing and infinite values replaced.
    """

    # First, replace infinite values with NaN 
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    if type == "zero":  # Replace NAs with zeroes
        for col in check_cols:
            df.loc[:, col] = df[col].fillna(0)
    elif type == "interpolate":  # Replace with linear interpolated values
        for col in check_cols:
            df.loc[:, col] = df[col].interpolate(method='linear') 
    elif type == "locf":  # Replace with last-observation-carried-forward
        for col in check_cols:
            df.loc[:, col] = df[col].fillna(method='ffill').fillna(method='bfill')
    elif type == "drop":
        df = df.dropna(subset=check_cols, axis=0).reset_index(drop=True)
    else:
        print("Warning: Invalid replacement 'type' specified. DataFrame returned unchanged.")
    
    # double check whether there is any NaN or infinite values in the DataFrame
    if df[check_cols].isnull().values.any():
        num_nan = df.isnull().sum().sum()
        #print(f"Warning: DataFrame still contains {num_nan} NaN values after replacement. Dropping rows with NaN values.")
        df = df.dropna(subset=check_cols, axis=0).reset_index(drop=True)

    return df

def _get_window_weights(window_type, window_length):
    """
    Helper function to generate normalized window weights.
    """
    if window_type == "flat":
        w = np.ones(window_length)
    else:
        # scipy.signal.windows provides various window functions
        # Use getattr for dynamic function call
        try:
            w = getattr(windows, window_type)(window_length)
        except AttributeError:
            raise ValueError(f"Unsupported window type: {window_type}")

    return w / np.sum(w) # Normalize the window

def _pad_signal(x, window_length, version):
    """
    Helper function to pad the signal according to the specified R version.
    This replicates the R code's specific reflection/symmetric padding.
    """
    N = len(x)
    
    if version == 1:
        # R: pre <- x[window_length:2]
        # Python: slice from window_length-1 down to 1 (inclusive), then reverse
        # This means elements from x[1] to x[window_length-1] in reverse order.
        pre = x[1:window_length][::-1]

        # R: post <- x[length(x):(length(x)-(window_length-2))]
        # Python: slice from N-1 down to N-(window_length-1) (inclusive), then reverse
        # This means elements from x[N-(window_length-1)] to x[N-1] in reverse order.
        post = x[N - (window_length - 1):N][::-1]
        
        s = np.concatenate((pre, x, post))
    elif version == 2:
        # R: pre <- 2 * x[1] - rev(x[2:window_length+1])
        # Python: 2 * x[0] - np.flip(x[1:window_length+1])
        # Note: R's x[2:window_length+1] means elements at index 2 up to window_length+1 (1-based).
        # In Python (0-based), this is x[1] up to x[window_length].
        pre = 2 * x[0] - np.flip(x[1:window_length + 1])

        # R: post <- 2 * x[length(x)] - x[length(x):(length(x)-(window_length-2))]
        # Python: 2 * x[-1] - np.flip(x[N - (window_length - 1):N])
        post = 2 * x[-1] - np.flip(x[N - (window_length - 1):N])
        
        s = np.concatenate((pre, x, post))
    else:
        raise ValueError("Invalid version. Must be 1 or 2.")
    
    return s

def _convolve_smooth(x, window_length=23, window="bartlett",version=1):
    """
    Smooths a signal using convolution with various windowing functions.
    Handles edge effects by introducing reflected copies of the signal.

    Args:
        x (Pandas Series representing a column of df): The input signal to be smoothed.
        window_length (int, optional): The dimension of the smoothing window.
                                       Should be an odd integer. Defaults to 23.
        window (str, optional): The type of window.
                                Can be 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
                                'flat' produces a moving average smoothing. Defaults to "bartlett".

    Returns:
        np.ndarray: The smoothed signal.

    Raises:
        ValueError: If input validation fails (e.g., signal too short, invalid window_length,
                    missing values, invalid window type, invalid version).
    """

    # Input validation
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float) # Ensure x is a numpy array

    if len(x) < window_length:
        raise ValueError("Input vector needs to be bigger than window length.")

    if window_length <= 2:
        raise ValueError("Window length must be odd and greater than 2.")

    if not np.all(np.isfinite(x)):
        num_missing = np.sum(~np.isfinite(x))
        raise ValueError(f"Missing values in data: 'x' contains {num_missing} missing values (NaN, Inf, or -Inf).")

    if window_length % 2 == 0:
        print("Warning: Window length should be an odd number for a symmetrical/centered rolling window.")
        window_length += 1

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.")

    N_x = len(x)
    original_window_length = window_length
    # Pad the signal 's' according to the specified version
    s = _pad_signal(x, window_length, version)
    
    # Generate normalized window weights
    w_norm = _get_window_weights(window, window_length)

    # Perform convolution using 'full' mode to match R's conv behavior before trimming
    y_full = convolve(s, w_norm, mode='full')

    # Trimming the convoluted signal to match the original R logic
    if version == 1:
        # R: y <- y[window_length:(length(y) - (window_length - 1))]
        # Python (0-indexed): y_full[start_idx : end_idx]
        # start_idx = window_length - 1
        # end_idx = -(window_length - 1) (using negative index for slice end)
        y_trimmed_step1 = y_full[window_length - 1 : -(window_length - 1)]

        # R: y <- y[(window_length %/% 2 + 1):(length(y) - (window_length %/% 2))]
        # Python (0-indexed): y_trimmed_step1[start_idx : end_idx]
        # k_trim_val_R = window_length // 2 # Integer division
        # start_idx = k_trim_val_R
        # end_idx = len(y_trimmed_step1) - k_trim_val_R
        k_trim_val_R = window_length // 2
        y = y_trimmed_step1[k_trim_val_R : len(y_trimmed_step1) - k_trim_val_R]

    elif version == 2:
        # R: y <- y[(window_length %/% 2 + 1):(length(y) - (window_length %/% 2))]
        k_trim_val_R = window_length // 2
        y_trimmed_step1 = y_full[k_trim_val_R : len(y_full) - k_trim_val_R]

        # R: y <- y[window_length:(length(y) - (window_length - 1))]
        y = y_trimmed_step1[window_length - 1 : -(window_length - 1)]
    
    # Final length check and adjustment (mimicking R's post-processing)
    
    if len(y) != N_x:
        print(f"Output length is not equal to input length: length(y): {len(y)} length(x): {N_x}\n"
                      "This happens when window_length is an even number (i.e. non-symmetric window around the center)\n")
        
        # Duplicate last y-value if original window length was even (to make input and output equal length)
        if original_window_length % 2 == 0:
            y = np.append(y, y[-1])
            # Re-check length after appending, as sometimes it might still not match due to complex trimming
            if len(y) != N_x:
                # This case indicates a more complex mismatch, which the R code's simple append might not fully fix
                # For exact replication, we'll ensure it's trimmed/padded to N_x
                if len(y) > N_x:
                    y = y[:N_x]
                elif len(y) < N_x:
                    # This scenario is less expected after the append if it was only off by 1
                    # but for robustness, pad with last value if still short
                    y = np.pad(y, (0, N_x - len(y)), mode='edge')
    return y

def savitzky_golay(y, window_length, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    if window_length % 2 != 1 or window_length < 1:
        raise TypeError("window_length size must be a positive odd number")
    if window_length < order + 2:
        raise TypeError("window_length is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_length -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def smooth_data(df, check_cols,type='convolve', window_type = 'bartlett', window_size=23):
    """
    Smooths the data in the DataFrame using either convolution or moving average or sav_golay.
    This function is called after removing blinks and NaNs to ensure the data is clean before smoothing.
    After removing nans, the columns in the DataFrame are desired and no need to select columns again.
    Args:
        df (pd.DataFrame): The input DataFrame containing data to be smoothed.
        type (str): The type of smoothing to apply ('convolve' , 'moving_average','sav_golay').
        window_size (int): The size of the smoothing window.

    Returns:
        pd.DataFrame: The smoothed DataFrame.
    """
    if type == 'convolve':
        df_subset = df[check_cols].apply(lambda x: _convolve_smooth(x.to_numpy(), window_length=window_size, window=window_type), axis=0)
    elif type == 'moving_average_mean':
        df_subset = df[check_cols].rolling(window=window_size, min_periods=1, center=True).mean()
    elif type == 'moving_average_median':
        df_subset = df[check_cols].rolling(window=window_size, min_periods=1, center=True).median()
    elif type == 'sav_golay':
        df_subset = df[check_cols].apply(lambda x: savitzky_golay(x, window_length=window_size, order=3), axis=0)
    else:
        print("Warning: Invalid smoothing 'type' specified. DataFrame returned unchanged.")
        return df
    #check whether df_subset has nan values
    if df_subset.isnull().values.any():
        num_nan = df_subset.isnull().sum().sum()
        print(f"Warning: DataFrame still contains {num_nan} NaN values after smoothing. Dropping rows with NaN values.")
    assert df_subset.shape[0] == df.shape[0], "The smoothed DataFrame must have the same number of rows as the original DataFrame."
    # replace the original columns with the smoothed columns
    df[check_cols] = df_subset
    return df
# TODO column name is hard code to tobii spectrum 
def scale_pixel_to_degrees(df,
                            screen_width_cm, screen_height_cm,
                            screen_resolution_width_px, screen_resolution_height_px,
                            assume_fixed_screen_distance=False,
                            center_normalize_gaze_coordinates=False):
    """
    Scales pixel coordinates of gaze data to degrees of visual angle.

    Args:
        df (pd.DataFrame): Input DataFrame containing gaze data. Expected columns include
                           'gaze_point_x', 'gaze_point_y', 'gaze_point_left_x', 'gaze_point_left_y',
                           'gaze_point_right_x', 'gaze_point_right_y',
                           'eye_position_z_dacs_mm' (for combined gaze),
                           'eye_position_left_z_dacs_mm', 'eye_position_right_z_dacs_mm'.
        screen_width_cm (float): Width of the screen in centimeters.
        screen_height_cm (float): Height of the screen in centimeters.
        screen_resolution_width_px (int): Resolution width of the screen in pixels.
        screen_resolution_height_px (int): Resolution height of the screen in pixels.
        assume_fixed_screen_distance (bool, optional): If True, uses the mean distance
                                                        to the screen as a constant for all samples.
                                                        Defaults to False.
        center_normalize_gaze_coordinates (bool, optional): If True, first normalizes
                                                             gaze coordinates to the screen center
                                                             before conversion. Defaults to False.

    Returns:
        pd.DataFrame: The input DataFrame with new columns for gaze coordinates in degrees of visual angle
                      ('gaze_point_x_deg', 'gaze_point_y_deg', etc.).
    """

    # Convert screen width in cm to mm
    screen_width_mm = screen_width_cm * 10
    screen_height_mm = screen_height_cm * 10

    if center_normalize_gaze_coordinates == False:
        # Convert pixels to degrees of visual angle
        if assume_fixed_screen_distance == False:  ## Each sample has its distance to screen
            df['gaze_point_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_x'] - screen_resolution_width_px / 2) /
                (df['eye_position_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_y'] - screen_resolution_height_px / 2) /
                (df['eye_position_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_left_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_left_x'] - screen_resolution_width_px / 2) /
                (df['eye_position_left_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_left_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_left_y'] - screen_resolution_height_px / 2) /
                (df['eye_position_left_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_right_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_right_x'] - screen_resolution_width_px / 2) /
                (df['eye_position_right_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_right_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_right_y'] - screen_resolution_height_px / 2) /
                (df['eye_position_right_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
        else:  ## Use the mean distance to screen as constant (averaged over both eyes)
            mean_screen_distance_mm = np.nanmean(np.concatenate([df['eye_position_left_z_dacs_mm'].dropna().values,
                                                                  df['eye_position_right_z_dacs_mm'].dropna().values]))
            df['gaze_point_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_x'] - screen_resolution_width_px / 2) /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_y'] - screen_resolution_height_px / 2) /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_left_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_left_x'] - screen_resolution_width_px / 2) /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_left_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_left_y'] - screen_resolution_height_px / 2) /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_right_x_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_right_x'] - screen_resolution_width_px / 2) /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_right_y_deg'] = (180 / np.pi) * np.arctan(
                (df['gaze_point_right_y'] - screen_resolution_height_px / 2) /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )

    else:

        # Get center normalized pixel coordinates
        gaze_point_x_center_normalized = df['gaze_point_x'] - (screen_resolution_width_px / 2)
        gaze_point_y_center_normalized = df['gaze_point_y'] - (screen_resolution_height_px / 2)
        gaze_point_left_x_center_normalized = df['gaze_point_left_x'] - (screen_resolution_width_px / 2)
        gaze_point_left_y_center_normalized = df['gaze_point_left_y'] - (screen_resolution_height_px / 2)
        gaze_point_right_x_center_normalized = df['gaze_point_right_x'] - (screen_resolution_width_px / 2)
        gaze_point_right_y_center_normalized = df['gaze_point_right_y'] - (screen_resolution_height_px / 2)

        # Convert pixels to degrees of visual angle
        if assume_fixed_screen_distance == False:  ## Each sample has its distance to screen
            df['gaze_point_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_x_center_normalized /
                (df['eye_position_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_y_center_normalized /
                (df['eye_position_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_left_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_left_x_center_normalized /
                (df['eye_position_left_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_left_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_left_y_center_normalized /
                (df['eye_position_left_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_right_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_right_x_center_normalized /
                (df['eye_position_right_z_dacs_mm'] * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_right_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_right_y_center_normalized /
                (df['eye_position_right_z_dacs_mm'] * screen_resolution_height_px / screen_height_mm)
            )
        else:  ## Use the mean distance to screen as constant (averaged over both eyes)
            mean_screen_distance_mm = np.nanmean(np.concatenate([df['eye_position_left_z_dacs_mm'].dropna().values,
                                                                  df['eye_position_right_z_dacs_mm'].dropna().values]))
            df['gaze_point_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_x_center_normalized /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            # Corrected division term for gaze_point_y_deg. Original R code had an error here.
            df['gaze_point_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_y_center_normalized /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_left_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_left_x_center_normalized /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            # Corrected division term for gaze_point_left_y_deg. Original R code had an error here.
            df['gaze_point_left_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_left_y_center_normalized /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )
            df['gaze_point_right_x_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_right_x_center_normalized /
                (mean_screen_distance_mm * screen_resolution_width_px / screen_width_mm)
            )
            df['gaze_point_right_y_deg'] = (180 / np.pi) * np.arctan(
                gaze_point_right_y_center_normalized /
                (mean_screen_distance_mm * screen_resolution_height_px / screen_height_mm)
            )

    return df

def downsample_data(array, sampling_rate, target_rate):
    """
    Downsamples the array to a target sampling rate.

    Args:
        array (numpy.array): The input array containing data to be downsampled.
        sampling_rate (int): The original sampling rate of the data.
        target_rate (int): The target sampling rate for downsampling.

    Returns:
        numpy.array: The downsampled array.
    """

    if target_rate >= sampling_rate:
        print("Target rate must be less than the original sampling rate. Returning original array.")
        return array

    if len(array)/15 < target_rate:
        #print("Current data is too short to downsample to the target rate. Returning original array.")
        return array
   
    # 1. Determine resampling parameters (up and down)
    # Find the simplest fraction for the ratio of new_freq / original_freq
    ratio_fraction = Fraction(int(target_rate * 1000), int(sampling_rate * 1000)).limit_denominator(1000)
    up = ratio_fraction.numerator
    down = ratio_fraction.denominator

    resampled_signal = resample_poly(array, up, down)

    return resampled_signal

def velocity_from_position(array, both, sampling_rate=1200):
    """
    Calculate velocity from position data in the array.
    This function computes the velocity by taking the gradient of the position data
    with respect to time, assuming a uniform sampling rate.
    Args:
        array: Input array containing position data.
        sampling_rate (int): Sampling rate of the data in Hz. Default is 1000Hz.

    Returns:
        numpy.array: Array with new columns for velocity.
    """

    window_length = 7  # Must be odd
    polyorder = 2
    dt = 1 / sampling_rate
    velocity=savgol_filter(array, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0)
    if both:
        return np.concatenate((array, velocity), axis=1)
    else:
        return velocity
    
def min_max_rescale(ts):
    #Preserves relative magnitudes
    ts_rescale = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-8)
    return ts_rescale

def z_score_standerdization(ts):
    #Centers data around 0 with unit variance
    ts_standardized = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
    return ts_standardized

def df_sanity_check(df,time_window):
    n_sample = len(df)
    remove_sample = np.mod(n_sample,time_window)
    # remove starting samples since the start of every trial may be the most noisy part.
    df = df.iloc[remove_sample:,:].reset_index(drop=True)
    return df


