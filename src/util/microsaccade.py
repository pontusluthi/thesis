import numpy as np

def vecvel(x, SAMPLING=1000, TYPE=2):
  N = x.shape[0] # Get number of rows
  v = np.zeros((N, 2)) # Initialize a 2-column array for velocities

  if TYPE == 2:
    # Central difference with 5-point stencil for internal points
    v[2:(N-2), :] = SAMPLING / 6 * (x[4:N, :] + x[3:(N-1), :] - x[1:(N-3), :] - x[0:(N-4), :])

    # For the second point (index 1 in Python, 2 in R)
    v[1, :] = SAMPLING / 2 * (x[2, :] - x[0, :])

    # For the second to last point (index N-2 in Python, N-1 in R)
    v[(N-2), :] = SAMPLING / 2 * (x[N-1, :] - x[(N-3), :])
  else:
    # Central difference with 3-point stencil for internal points
    v[1:(N-1), :] = SAMPLING / 2 * (x[2:N, :] - x[0:(N-2), :])
  return v

def _microsac_features(sac_list,v,x):
    """Compute microsaccade features from a list of saccades.
    sac_table
            Array with shape (num_saccades, 7) containing saccade features:
            [onset, end, 
            peak velocity, 
            horizontal component, vertical component, 
            horizontal amplitude, vertical amplitude]
    """
    sac_table = np.array(sac_list,dtype=np.float32) # num_sac*feature_dim
    nsac = sac_table.shape[0] # Number of saccades
    # Compute peak velocity, horizontal and vertical components
    for s in range(nsac):
        # Onset and offset for saccades (converted back to original time points)
        onset = int(sac_table[s, 0])
        offset = int(sac_table[s, 1])
        idx = np.arange(onset, offset + 1) # Inclusive range

        # Saccade peak velocity (vpeak)
        vpeak = np.max(np.sqrt(v[idx, 0]**2 + v[idx, 1]**2))
        sac_table[s, 2] = vpeak

        # Saccade vector (dx,dy) horizontal and vertical components
        dx = x[offset, 0] - x[onset, 0]
        dy = x[offset, 1] - x[onset, 1]
        sac_table[s, 3:5] = [dx, dy]

        # Saccade amplitude (dX,dY) horizontal and vertical amplitudes
        minx = np.min(x[idx, 0])
        maxx = np.max(x[idx, 0])
        miny = np.min(x[idx, 1])
        maxy = np.max(x[idx, 1])

        ix1 = np.argmin(x[idx, 0]) # Get index within idx that corresponds to minx
        ix2 = np.argmax(x[idx, 0]) # Get index within idx that corresponds to maxx
        iy1 = np.argmin(x[idx, 1]) # Get index within idx that corresponds to miny
        iy2 = np.argmax(x[idx, 1])

        dX = np.sign(ix2 - ix1) * (maxx - minx)
        dY = np.sign(iy2 - iy1) * (maxy - miny)
        sac_table[s, 5:7] = [dX, dY]
    return sac_table

def microsac_final_features(sac_array):
    """Compute final microsaccade features from sac_array.
        #  Basic saccade parameters: (0) onset, (1) end, (2) duration, 
        #   (3) peak velocity, (4) distance, 
        #  (5) orientation related to distance vector, (6) amplitude, 
        #  (7) orientation related to amplitude vector
    """
    sac_table = np.zeros((sac_array.shape[0], 8))
    # Onset and offset
    sac_table[:, 0:2] = sac_array[:, 0:2]
    # Duration
    sac_table[:, 2] = sac_array[:, 1] - sac_array[:, 0] + 1
    # Peak velocity
    sac_table[:, 3] = sac_array[:, 2]
    # Distance
    sac_table[:, 4] = np.sqrt(sac_array[:, 3]**2 + sac_array[:, 4]**2)
    # Orientation of distance
    sac_table[:, 5] = np.arctan2(sac_array[:, 4], sac_array[:, 3]) #radius 
    # Amplitude
    sac_table[:, 6] = np.sqrt(sac_array[:, 5]**2 + sac_array[:, 6]**2)
    # Orientation of amplitude
    sac_table[:, 7] = np.arctan2(sac_array[:, 6], sac_array[:, 5])
    return sac_table

def microsacc(x,v, VFAC=5, MINDUR=3, SAMPLING=500):
    """Detect microsaccades in a velocity vector for one eye
    Parameters:
    ----------
    x : np.ndarray
        Position vector with shape (N, 2) where N is the number of samples.
    v : np.ndarray
        Velocity vector with shape (N, 2) where N is the number of samples.
    VFAC : float
        Relative velocity threshold.
    MINDUR : int
        Minimum duration of a microsaccade in samples.
    SAMPLING : int
        Sampling rate in Hz.
    Returns:
    -------
    sac : dict or None
        Dictionary containing saccade information or None if no saccades are detected.
        {
        "data": sac_array
                Array with shape (num_saccades, 9) containing saccade features:
                Basic saccade parameters: (1) onset, (2) end, (3) duration, 
                (4) delay between eyes, (5) peak velocity, (6) distance, 
                (7) orientation related to distance vector, (8) amplitude, 
                (9) orientation related to amplitude vector

        "radius": radius
                  Parameters of elliptic threshold
        }
    """
    if v.size==0:
        v = vecvel(x, SAMPLING, TYPE=2)  # Calculate velocity if not provided
    # Compute threshold
    medx = np.median(v[:, 0])
    msdx = np.sqrt(np.median((v[:, 0] - medx)**2))
    medy = np.median(v[:, 1])
    msdy = np.sqrt(np.median((v[:, 1] - medy)**2))

    if msdx < 1e-10:
        msdx = np.sqrt(np.mean(v[:, 0]**2) - (np.mean(v[:, 0]))**2)
    if msdy < 1e-10:
        msdy = np.sqrt(np.mean(v[:, 1]**2) - (np.mean(v[:, 1]))**2)

    radiusx = VFAC * msdx
    radiusy = VFAC * msdy
    radius = np.array([radiusx, radiusy])

    # Apply test criterion: elliptic threshold
    test = (v[:, 0] / radiusx)**2 + (v[:, 1] / radiusy)**2
    indx = np.where(test > 1)[0] # sample number that exceeds the threshold

    # Determine saccades
    N = len(indx)
    nsac = 0
    sac_list = []
    dur = 1
    start_idx = 0 # Index in indx array for the start of a potential saccade
    k = 0

    # Loop over saccade candidates
    while k < N:
        if k + 1 < N and indx[k + 1] - indx[k] == 1:
            dur += 1
        else:
        # Minimum duration criterion
            if dur >= MINDUR:
                nsac += 1
                end_idx = k
                # Store as [onset, end, 0,0,0,0,0]
                sac_list.append([indx[start_idx], indx[end_idx], 0, 0, 0, 0, 0])
            start_idx = k + 1
            dur = 1
        k += 1

    # Check minimum duration for last microsaccade
    if dur >= MINDUR:
        nsac += 1
        end_idx = k - 1 # k is already incremented in the loop
        sac_list.append([indx[start_idx], indx[end_idx], 0, 0, 0, 0, 0])

    # Postprocess microsaccades
    postprocess_result = postprocess(sac_list, PP_TYPE="merge", MIN_TEMP_DIST_MS=10, SAMPLING=SAMPLING)
    sac_list = postprocess_result['sac_merged']

    if nsac > 0:
        sac_array = _microsac_features(sac_list,v,x)
        
        if sac_array is None or len(sac_array) == 0:
            return {'data': np.empty((0, 8)), 'valid': False}
        
        sac_table = microsac_final_features(sac_array) 
        # abnormal if col 3 > 150 OR col 6 > 1.75
        mask_abnormal = (sac_table[:, 3] > 150) | (sac_table[:, 6] > 1.75)
        # keep only normal rows (exclude abnormal)
        sac_table_clean = sac_table[~mask_abnormal]
        sac = {'data': sac_table_clean, 'radius': radius}
    else:
        sac = None
    return sac

def microsacc_PDRT_event(x,v, idx_sac,MINDUR=3):
    """
    Extract microsaccades and their features from position and velocity data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing position and velocity data.
    VFAC (float): Relative velocity threshold.
    MINDUR (int): Minimum duration of microsaccades in samples.
    SAMPLING (int): Sampling rate in Hz.
    
    Returns:
    dict: Dictionary containing microsaccade data and radius parameters.
    """
    #consecutive indices indicates a microsaccade,get the start and end index of each microsaccade
    idx_sac = np.array(idx_sac)
    idx_sac_diff = np.diff(idx_sac)
    sac_list = []
    start_idx = 0
    for i in range(len(idx_sac_diff)):
        if idx_sac_diff[i] > 1:
            if i - start_idx > MINDUR:
                sac_list.append([idx_sac[start_idx], idx_sac[i],0, 0, 0, 0, 0])
            start_idx = i + 1
    if start_idx < len(idx_sac):
        if idx_sac[-1] - idx_sac[start_idx]+1 > MINDUR:
            sac_list.append([idx_sac[start_idx], idx_sac[-1],0, 0, 0, 0, 0])
    #start and end index of microsaccade retrieved from the event column

    sac = _microsac_features(sac_list,v,x)
    sac = microsac_final_features(sac)
    return sac

def binsacc(sacl, sacr):
    """
    Determine binocular and monocular saccades from left and right eye saccades.
    """
    numr = sacr.shape[0] if sacr is not None else 0
    numl = sacl.shape[0] if sacl is not None else 0
    NB = 0
    NR = 0
    NL = 0
    bin_saccades = []
    monol_saccades = []
    monor_saccades = []

    if numr * numl > 0:
        # Determine saccade clusters 
        # Create a time series of saccade onsets and offsets
        TR = np.max(sacr[:, 1]) # R's column 2 is Python's index 1
        TL = np.max(sacl[:, 1])
        TB = int(max(TL, TR))
        s = np.zeros(TB + 1) # add an extra element for diff

        for i in range(numl):
            left_onset = int(sacl[i, 0])
            left_offset = int(sacl[i, 1])
            s[left_onset : left_offset + 1] = 1

        for i in range(numr):
            right_onset = int(sacr[i, 0])
            right_offset = int(sacr[i, 1])
            s[right_onset : right_offset + 1] = 1

        s[0] = 0
        s[TB] = 0 # Ensure last element is 0 for diff

        # Find onsets and offsets of microsaccades
        onoff = np.where(np.diff(s) != 0)[0].reshape(-1, 2) # Find indices where the difference is non-zero, Reshape into 2-column matrix, row-wise. Each element is [start,end]
        N = onoff.shape[0] # onoff shape: num_sac*2 where 2 is for onset and offset

        # Determine binocular saccades TODO following right or wrong?
        for i in range(N):
            current_onset = onoff[i, 0]
            current_offset = onoff[i, 1]

            left_indices = np.where((current_onset <= sacl[:, 0]) & (sacl[:, 1] <= current_offset))[0]
            right_indices = np.where((current_onset <= sacr[:, 0]) & (sacr[:, 1] <= current_offset))[0]

            # Binocular saccades
            if len(left_indices) > 0 and len(right_indices) > 0:
                ampr = np.sqrt(sacr[right_indices, 5]**2 + sacr[right_indices, 6]**2) # R's column 6,7 is Python's index 5,6
                ampl = np.sqrt(sacl[left_indices, 5]**2 + sacl[left_indices, 6]**2)

                # Determine largest event in each eye
                ir = right_indices[np.argmax(ampr)]
                il = left_indices[np.argmax(ampl)]
                NB += 1
                bin_saccades.append(np.concatenate((sacl[il, :], sacr[ir, :])))
            else:
                # Determine monocular saccades
                if len(left_indices) == 0 and len(right_indices) > 0:
                    NR += 1
                    ampr = np.sqrt(sacr[right_indices, 5]**2 + sacr[right_indices, 6]**2)
                    ir = right_indices[np.argmax(ampr)]
                    monor_saccades.append(sacr[ir, :])
                elif len(right_indices) == 0 and len(left_indices) > 0:
                    NL += 1
                    ampl = np.sqrt(sacl[left_indices, 5]**2 + sacl[left_indices, 6]**2)
                    il = left_indices[np.argmax(ampl)]
                    monol_saccades.append(sacl[il, :])
    else:
        # Special case of exclusively monocular saccades
        if numr == 0:
            bin_saccades = []
            monor_saccades = []
            monol_saccades = sacl.tolist() if sacl is not None else []
        if numl == 0:
            bin_saccades = []
            monol_saccades = []
            monor_saccades = sacr.tolist() if sacr is not None else []

    sac = {
        'N': np.array([NB, NL, NR]),
        'bin': np.array(bin_saccades) if bin_saccades else None,
        'monol': np.array(monol_saccades) if monol_saccades else None,
        'monor': np.array(monor_saccades) if monor_saccades else None
    }
    return sac

def sacpar(sac):
  M = sac['N'][0] # Access the first element of the 'N' array
  if M < 1:
    sacs = np.array([])
  else:
    # 1. Onset
    a = np.column_stack((sac['bin'][:, 0], sac['bin'][:, 7])) # R's column 1 is Python's index 0
    a_min = np.min(a, axis=1)

    # 2. Offset
    b = np.column_stack((sac['bin'][:, 1], sac['bin'][:, 8])) # R's column 2 is Python's index 1
    b_min = np.min(b, axis=1)

    # 3. Duration
    DR = sac['bin'][:, 1] - sac['bin'][:, 0] + 1
    DL = sac['bin'][:, 8] - sac['bin'][:, 7] + 1
    D = (DR + DL) / 2

    # 4. Delay between eyes
    delay = b_min - a_min + 1

    # 5. Peak velocity
    vpeak = (sac['bin'][:, 2] + sac['bin'][:, 9]) / 2

    # 6. Saccade distance
    dist = (np.sqrt(sac['bin'][:, 3]**2 + sac['bin'][:, 4]**2) +
            np.sqrt(sac['bin'][:, 10]**2 + sac['bin'][:, 11]**2)) / 2
    angle1 = np.arctan2((sac['bin'][:, 4] + sac['bin'][:, 11]) / 2,
                        (sac['bin'][:, 3] + sac['bin'][:, 10]) / 2)

    # 7. Saccade amplitude
    ampl = (np.sqrt(sac['bin'][:, 5]**2 + sac['bin'][:, 6]**2) +
            np.sqrt(sac['bin'][:, 12]**2 + sac['bin'][:, 13]**2)) / 2
    angle2 = np.arctan2((sac['bin'][:, 6] + sac['bin'][:, 13]) / 2,
                        (sac['bin'][:, 5] + sac['bin'][:, 12]) / 2)

    sacs = np.zeros((M, 9))
    sacs[:, 0] = a_min
    sacs[:, 1] = b_min
    sacs[:, 2] = D
    sacs[:, 3] = delay
    sacs[:, 4] = vpeak
    sacs[:, 5] = dist
    sacs[:, 6] = angle1
    sacs[:, 7] = ampl
    sacs[:, 8] = angle2
  return sacs


def duration_to_samples(duration_ms, sampling_rate):
    """
    Convert duration in milliseconds to number of samples.
    
    Parameters:
    -----------
    duration_ms : float
        Duration in milliseconds
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    int : Number of samples corresponding to the duration
    """
    return int(duration_ms * sampling_rate / 1000)

def find_terminating_saccade(sac_idx, sac, MIN_TEMP_DIST):
    """
    Find the saccade that terminates the input saccade.
    
    Parameters:
    -----------
    sac_idx : int
        Index of the current saccade
    sac : list
        list of saccades with shape (num_saccades, 7)
    MIN_TEMP_DIST : int
        Minimum temporal distance in samples
        
    Returns:
    --------
    int : Index of the terminating saccade
    """
    candidate_offset = sac[sac_idx][1]  # end time of current saccade
    terminating_saccade = None
    test = False
    
    if sac_idx == len(sac) - 1:
        terminating_saccade = sac_idx
    else:
        while not test:
            dist = sac[sac_idx + 1][0] - candidate_offset  # distance to next saccade onset
            if dist > MIN_TEMP_DIST:
                terminating_saccade = sac_idx
                test = True
            elif dist <= MIN_TEMP_DIST:
                candidate_offset = sac[sac_idx + 1][1]  # update to next saccade end
                sac_idx = sac_idx + 1
                if sac_idx == len(sac) - 1:
                    terminating_saccade = sac_idx
                    test = True
    
    return terminating_saccade

def merge_saccades(sac, MIN_TEMP_DIST_MS, SAMPLING):
    """
    Merge consecutive saccades close in time.
    
    Parameters:
    -----------
    sac : np.ndarray
        Array of saccades with shape (num_saccades, 7)
    MIN_TEMP_DIST_MS : float
        Minimum temporal distance in milliseconds
    SAMPLING : int
        Sampling rate in Hz
        
    Returns:
    --------
    dict : Dictionary containing merged saccades and segment counts
    """
    # Convert MIN_TEMP_DIST_MS to integer number of samples
    MIN_TEMP_DIST = duration_to_samples(MIN_TEMP_DIST_MS, SAMPLING)
    
    # Check that MIN_TEMP_DIST is an integer
    if MIN_TEMP_DIST % 1 != 0:
        raise ValueError("MIN_TEMP_DIST is not an integer")
    
    this_sac_idx = 0
    terminating_saccade_idx = 0
    sac_cnt = 1
    sac_merged = []
    n_micsac_segments_v = []
    
    while terminating_saccade_idx < len(sac)-1:
        terminating_saccade_idx = find_terminating_saccade(this_sac_idx, sac, MIN_TEMP_DIST)
        n_micsac_segments = (terminating_saccade_idx - this_sac_idx) + 1
        
        # Create merged saccade: [onset, end, 0, 0, 0, 0, 0]
        merged_saccade = [sac[this_sac_idx][0], sac[terminating_saccade_idx][1]] + [0] * 5
        sac_merged.append(merged_saccade)
        
        this_sac_idx = terminating_saccade_idx + 1
        sac_cnt += 1
        
        # Validation check
        #if n_micsac_segments > 5:
        #    print("More than five saccades were merged together: you may want to double check this.")
        
        n_micsac_segments_v.append(n_micsac_segments)
    
    return {
        'sac_merged': np.array(sac_merged),
        'n_micsac_segments_v': np.array(n_micsac_segments_v)
    }

def discard_saccades(sac, MIN_TEMP_DIST_MS, SAMPLING):
    """
    Discard consecutive saccades close in time (keep only the first).
    
    Parameters:
    -----------
    sac : np.ndarray
        Array of saccades with shape (num_saccades, 7)
    MIN_TEMP_DIST_MS : float
        Minimum temporal distance in milliseconds
    SAMPLING : int
        Sampling rate in Hz
        
    Returns:
    --------
    dict : Dictionary containing discarded saccades and segment counts
    """
    #print("Discarding saccades close in time (keeping the first one only)")
    
    # Convert MIN_TEMP_DIST_MS to integer number of samples
    MIN_TEMP_DIST = duration_to_samples(MIN_TEMP_DIST_MS, SAMPLING)
    #print(f"MIN_TEMP_DIST (in number of samples): {MIN_TEMP_DIST}")
    
    # Check that MIN_TEMP_DIST is an integer
    if MIN_TEMP_DIST % 1 != 0:
        raise ValueError("MIN_TEMP_DIST is not an integer")
    
    this_sac_idx = 0
    terminating_saccade_idx = 0
    sac_cnt = 1
    sac_discarded = []
    n_micsac_segments_v = []
    
    while terminating_saccade_idx < len(sac):
        terminating_saccade_idx = find_terminating_saccade(this_sac_idx, sac, MIN_TEMP_DIST)
        
        # Keep only the first saccade in the sequence
        sac_discarded.append(sac[this_sac_idx].copy())
        this_sac_idx = terminating_saccade_idx + 1
        sac_cnt += 1
        n_micsac_segments_v.append(1)
        
        # Validation check
        if (terminating_saccade_idx - this_sac_idx) > 3:
            #print(f"{sac_cnt}: {this_sac_idx} - {terminating_saccade_idx}")
            raise ValueError("More than three saccades were discarded: you may want to double check this.")
    
    return {
        'sac_discarded': np.array(sac_discarded),
        'n_micsac_segments_v': np.array(n_micsac_segments_v)
    }

def postprocess(sac, PP_TYPE="merge", MIN_TEMP_DIST_MS=10, SAMPLING=600):
    """
    Postprocess monocularly detected saccades using one of two options:
    1: Merge saccades close in time
    2: Discard saccades close in time (keep only the first)
    
    Parameters:
    -----------
    sac : np.ndarray
        Array of saccades with shape (num_saccades, 7) containing:
        [onset, end, peak_velocity, horizontal_component, vertical_component, 
         horizontal_amplitude, vertical_amplitude]
    PP_TYPE : str
        "merge", "discard", or "none"
    MIN_TEMP_DIST_MS : float
        Minimal temporal distance in milliseconds. The time that must elapse between 
        consecutive saccades in order for them to be treated as separate saccades.
    SAMPLING : int
        Sampling rate in Hz
        
    Returns:
    --------
    dict : Dictionary containing processed saccades and metadata
    """
    #("Running postprocess()")
    #print(f"PP_TYPE: {PP_TYPE}")
    #print(f"MIN_TEMP_DIST_MS: {MIN_TEMP_DIST_MS}")
    #print(f"SAMPLING: {SAMPLING}")
    #print()
    
    if PP_TYPE == "merge":
        #print(f"Number of microsaccades prior to merging saccades close in time: {len(sac)}")
        result = merge_saccades(sac, MIN_TEMP_DIST_MS, SAMPLING)
        #print(f"Number of microsaccades after merging saccades close in time: {len(result['sac_merged'])}")
        return result
        
    elif PP_TYPE == "discard":
        #print(f"Number of microsaccades prior to discarding saccades close in time (keeping only the first instance): {len(sac)}")
        result = discard_saccades(sac, MIN_TEMP_DIST_MS, SAMPLING)
        #print(f"Number of microsaccades after discarding saccades close in time (Keeping only the first instance): {len(result['sac_discarded'])}")
        return result
        
    elif PP_TYPE == "none":
        #print("No postprocessing applied. Returning input saccades.")
        n_micsac_segments_v = np.ones(len(sac))
        return {
            'sac': sac,
            'n_micsac_segments_v': n_micsac_segments_v
        }
    
    else:
        raise ValueError(f"Invalid PP_TYPE: {PP_TYPE}. Must be 'merge', 'discard', or 'none'")
def microsaccade_extraction(velocity, position, sampling_rate, VFAC=5, MINDUR=3):
    """
    Extract microsaccades from the dataframe based on the velocity threshold.
    
    Parameters:
    velocity (np.ndarray): Array containing velocity data. Can be empty.
    position (np.ndarray): Array containing position data.
    VFAC (float): Relative velocity threshold.
    MINDUR (int): Minimum duration of microsaccades in samples.
    Returns:
    pd.DataFrame: DataFrame containing extracted microsaccades.
    """
    sac = microsacc(position, velocity, VFAC=VFAC, MINDUR=MINDUR, SAMPLING=sampling_rate)
    microsaccades = sac['data'] if sac is not None else []
    return microsaccades