# Functions for the frequency tagging condition of the experiment 2
import scipy
import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, cross_val_multiscore

def coherence_kabir(signalX, pick, freq_of_interest):

    #get info from EEG
    min_time = signalX.times[0]
    max_time = signalX.times[-1]
    sampling_rate = signalX.info['sfreq']
    n = len(signalX)  # number of trials

    # Band-pass EEG (+/-1.9Hz) and apply hilbert
    signalX = signalX.copy().pick(pick).filter(l_freq=freq_of_interest - 1.9, h_freq=freq_of_interest + 1.9,
        method='iir', iir_params=dict(order=4, ftype='butter'), verbose = False).get_data(copy=False)
    signalX = np.squeeze(signalX).T
    signalXh = scipy.signal.hilbert(signalX, axis=0)

    #Create sine wave
    t = np.linspace(min_time, max_time, int(sampling_rate* (np.abs(min_time) + max_time)+1))
    signalY = np.sin(2 * np.pi * freq_of_interest * t)
    signalY = np.tile(signalY, (n,1)) #repeat over trials
    
    # Hilbert transform
    signalYh = scipy.signal.hilbert(signalY.T, axis=0)

    # Magnitude
    mX = np.abs(signalXh).T
    mY = np.abs(signalYh).T

    # Phase difference
    phase_diff = np.angle(signalXh).T - np.angle(signalYh).T
    
    coh = np.zeros(signalY.shape[1])
    for t in range(signalY.shape[1]):
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


def ssvep_amplitudes(epochs, electrodes, tmin, tmax):
    '''
    For each condition in queries, each trial and electrode, return the complex Fourier coefficient

    based on the method in in Chota, Bruat, Stigchel & Strauch 2023
    '''
    
    # Calculate FFT over trial-averaged signal
    data = epochs.copy().crop(tmin=tmin, tmax=tmax).pick(electrodes).average().get_data()

    # Zero-pad the data to increase freq resolution and faster computation (power of 2)
    padded_data = np.zeros((len(electrodes), 2**15))
    padded_data[:, :data.shape[1]] = data
    # For all electrodes
    N = padded_data.shape[-1]
    fft_values = np.fft.fft(padded_data, axis=-1)
    fft_freq = np.fft.fftfreq(N, 1/epochs.info['sfreq'])
    
    # Only keep the positive frequencies (fourier coefficients)
    positive_freq_indices = np.where(fft_freq >= 0)
    fft_freq = fft_freq[positive_freq_indices]
    fft_values = fft_values[:,positive_freq_indices]
    
    # Compute the magnitude of the FFT
    fft_magnitude = np.abs(fft_values)**2

    return fft_freq, fft_magnitude.squeeze()

def model_training(epochs, tmin_training,tmax_training,tmin_prediction,tmax_prediction):
    '''
    Function to train on a given period and test on another.
    tmax_training and tmin_training are used for training the model
    tmax_prediciton and tmin_prediciton are used for prediction only based on the best training time
    '''
    epochs_posttraining = epochs.copy().crop(tmin=tmin_training, tmax=tmax_training)
    epochs_pretraining = epochs.copy().crop(tmin=tmin_prediction, tmax=tmax_prediction ).copy()
    
    # Data and label to train on (predicting side that was cued)
    X = epochs_posttraining.copy().get_data()
    y = epochs_posttraining.metadata['cued'].values
    
    # Model fitting with logistic regression, using 5 CV
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    time_decod = SlidingEstimator(clf, n_jobs=5, scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=5)
    
    # Find the best time point during training period, first average scores over splits and pick max score
    mean_scores = np.mean(scores, axis=0)
    best_time_idx = np.argmax(mean_scores)
    best_time = epochs_posttraining.times[best_time_idx]
    best_score = mean_scores[best_time_idx]
    print(f"Best time point post-250ms: {best_time} seconds with score: {best_score}")
    
    # Re-train the model on the best time during the training period only
    best_time_data = X[:, :, best_time_idx]
    clf_best = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    clf_best.fit(best_time_data, y)
    
    # Evaluate the model on data for each time point during prediction period separately for endo and exo
    exo = epochs_pretraining['condition=="exo"'].copy()
    endo = epochs_pretraining['condition=="endo"'].copy()
    # Automatically perform kfold
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores_pretraining_exo = []
    cv_scores_pretraining_endo = []
    
    # Iterate through each time point in the prediction period
    for t in range(len(exo.times)):
        X_pretraining_exo = exo.get_data()[:, :, t]
        X_pretraining_endo = endo.get_data()[:, :, t]
        y_exo = exo.metadata['cued'].values
        y_endo = endo.metadata['cued'].values
        fold_scores_exo = []
        fold_scores_endo = []
        
        # Iterate through each fold of cross-validation
        for train_index, test_index in skf.split(X_pretraining_exo, y_exo):
            X_train, X_test = X_pretraining_exo[train_index], X_pretraining_exo[test_index]
            y_train, y_test = y_exo[train_index], y_exo[test_index]
            
            # Use the previously trained model to predict the time point data in the prediction period
            predictions = clf_best.predict(X_pretraining_exo[test_index])
            score_exo = roc_auc_score(y_test, predictions)
            fold_scores_exo.append(score_exo)
        # Average score across folds for this time point
        avg_score_exo = np.mean(fold_scores_exo, axis=-1)
        cv_scores_pretraining_exo.append(avg_score_exo)
        
        # Same for endo
        for train_index, test_index in skf.split(X_pretraining_endo, y_endo):
            X_train, X_test = X_pretraining_endo[train_index], X_pretraining_endo[test_index]
            y_train, y_test = y_endo[train_index], y_endo[test_index]
            
            predictions = clf_best.predict(X_pretraining_endo[test_index])
            score_endo = roc_auc_score(y_test, predictions)
            fold_scores_endo.append(score_endo)
        
        avg_score_endo = np.mean(fold_scores_endo, axis=-1)
        cv_scores_pretraining_endo.append(avg_score_endo)

    cv_scores_pretraining_exo = np.array(cv_scores_pretraining_exo)
    cv_scores_pretraining_endo = np.array(cv_scores_pretraining_endo)
    
    # Make all scores > .50 as there are only two labels
    cv_scores_pretraining_exo = np.abs(cv_scores_pretraining_exo - .5)+.5
    cv_scores_pretraining_endo = np.abs(cv_scores_pretraining_endo - .5)+.5

    pd_prediction = pd.DataFrame({'exo':cv_scores_pretraining_exo,
                     'endo':cv_scores_pretraining_endo,
                     'time':epochs_pretraining.times})
    pd_prediction = pd_prediction.melt(id_vars='time')
    pd_prediction.columns = ['time', 'attention', 'score']
    pd_prediction['best_time'] = best_time
    return pd_prediction, clf_best
