# Functions for the frequency tagging condition of the experiment 1 
import scipy
import numpy as np

def coherence_kabir(signalX, pick, freq_of_interest):

    #get info from EEG
    min_time = signalX.times[0]
    max_time = signalX.times[-1]
    sampling_rate = signalX.info['sfreq']
    
    # Band-pass EEG (+/-1.9Hz) and apply hilbert
    signalX = signalX.copy().pick(pick).filter(l_freq=freq_of_interest - .5, h_freq=freq_of_interest + .5,
        method='iir', iir_params=dict(order=4, ftype='butter'), phase='zero', fir_window='hamming', verbose = False)
    #filter(l_freq = freq_of_interest - 1.9, h_freq = freq_of_interest + 1.9, verbose=True)
    
    signalX = np.squeeze(signalX.get_data(copy=False)).T
    signalXh =  scipy.signal.hilbert(signalX, axis=1)
    n = signalXh.shape[1]  # number of trials

    #Create sine wave
    t = np.linspace(min_time, max_time, int(sampling_rate * (np.abs(min_time) + max_time))+1, endpoint=False)
    signalY = np.sin(2 * np.pi * freq_of_interest * t)
    signalY = np.tile(signalY, (n,1)).T #repeat over trials
    # Hilbert transform
    signalYh = scipy.signal.hilbert(signalY.T, axis=1)

    # Magnitude
    mX = np.abs(signalXh).T
    mY = np.abs(signalYh)

    # Phase difference
    phase_diff = np.angle(signalXh).T - np.angle(signalYh)

    coh = np.zeros(signalY.shape[0])
    for t in range(signalY.shape[0]):
        num = ((np.abs(np.sum(mX[:, t] * mY[:, t] * np.exp(1j * phase_diff[:, t])) / n)) ** 2)
        denom = (np.sum((mX[:, t]**2) * (mY[:, t]**2)) / n)
        coh[t] = num/denom
        
    return coh

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """
    # Signal to noise ratio (Meigen & Bach (1999))
    from https://mne.tools/dev/auto_tutorials/time-freq/50_ssvep.html
    Compute SNR spectrum from PSD spectrum using convolution.
    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


def ssvep_amplitudes(epochs, queries, frequencies, electrodes, tmin, tmax):
    '''
    For each condition in queries, each trial and electrode, return the complex Fourier coefficient
    '''
    n_points = 2**14
    ssvep_amp = {cond: [] for cond in queries}
    
    for condition in queries:
        # Calculate FFT over trial-averaged signal
        data =  epochs[condition].copy().crop(tmin=tmin, tmax=tmax).pick(electrodes).average().get_data()
        
        # Zero-pad the data to increase freq resolution and faster computation (power of 2)
        padded_data = np.zeros((len(electrodes), n_points))
        padded_data[:, :data.shape[1]] = data
        
        # Compute FFT
        fft_results = np.fft.fft(padded_data, axis=1)
        
        # Calculate the amplitude of the complex Fourier coefficients
        amplitudes = np.abs(fft_results)
        
        # Store the amplitudes for each frequency of interest
        all_freqs = []
        for i, freq in enumerate(frequencies):
            freq_index = int(freq * n_points / epochs.info['sfreq'])
            all_freqs.append(amplitudes[:, freq_index])
        
        ssvep_amp[condition].append(np.array(all_freqs).T)

    # Convert lists to arrays
    for condition in queries:
             ssvep_amp[condition] = np.array(ssvep_amp[condition][0])
    return ssvep_amp


def frequency_rescaling(A):
    '''
    Rescale frequency for statistical analysis as in Adamian & Andersen, 2024
    object is PSD, shape is frequency, cueing condition
    '''
    normalized_Ajk = np.zeros(A.shape)
    for electrode in range(A.shape[1]):
        # Average across cueing condition
        mean_Ajk = np.mean(A[:,electrode])
    
        # Divide each amplitude Ajk by the mean frequency for all cueing condition
        normalized_Ajk[:,electrode] = A[:,electrode] / mean_Ajk

    return normalized_Ajk