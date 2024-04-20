import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import resample
from biosppy.signals import ecg
from scipy.signal import find_peaks
import re
import os
import pywt
from sklearn.preprocessing import MinMaxScaler


def filter_row(row):
    # Design a low-pass Butterworth filter
    b, a = signal.butter(2, 30 / (500 / 2), btype='low')

    # Apply the filter to the ECG signal
    filtered_ecg = signal.filtfilt(b, a, row)
    
    # Apply high-pass filtering to remove baseline wander (cut-off frequency = 0.5 Hz)
    b, a = signal.butter(2, 1/(500/2), 'high')
    filtered_ecg = signal.filtfilt(b, a, filtered_ecg)

    # Apply notch filtering to remove powerline noise (50 Hz) and its harmonics
    f0 = 50.0  # Frequency to be removed from the signal
    Q = 30.0   # Quality factor
    w0 = f0/(500/2)  # Normalized frequency
    b, a = signal.iirnotch(w0, Q)
    filtered_ecg = signal.filtfilt(b, a, filtered_ecg)
    filtered_ecg = filtered_ecg/max(abs(filtered_ecg))
    
    return filtered_ecg

def downsample_row(row):

    # Downsample the signal
    resampled_signal = resample(row, int(len(row) * 125 / 500))

    return resampled_signal

def extract_ecg_features(ecg_signal, sampling_rate = 125):
    """
    Extract ECG features including HRV and durations of P-wave, QRS complex, T-wave, QT interval, and PR interval.

    Parameters:
    ecg_signal (numpy.ndarray): The ECG signal.
    sampling_rate (int): Sampling rate of the ECG signal (Hz).

    Returns:
    pandas.DataFrame: DataFrame containing extracted ECG features.
    """
    ecg_signal = ecg_signal / max(abs(ecg_signal))

    # Extract R peaks using biosppy
    ecg_analysis = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    rpeaks = ecg_analysis['rpeaks']

    # Compute other features
    pr_intervals = []
    qt_intervals = []
    p_wave_durations = []
    qrs_durations = []
    t_wave_durations = []

    for i in range(len(rpeaks) - 1):
        # P-wave detection
        p_wave_start = rpeaks[i] - 30
        p_wave_peak, _ = find_peaks(ecg_signal[p_wave_start:rpeaks[i]], height=[0, 0.5])
        if len(p_wave_peak) == 0:
            continue
        p_wave_peak = p_wave_start + p_wave_peak[0]
        p_wave_end = rpeaks[i]
        p_wave_duration = p_wave_end - p_wave_peak
        p_wave_durations.append(p_wave_duration/sampling_rate)

        # QRS complex duration
        qrs_duration = rpeaks[i + 1] - rpeaks[i]
        qrs_durations.append(qrs_duration/sampling_rate)

        # T-wave detection
        t_wave_end = rpeaks[i + 1] - 30
        t_wave_peak, _ = find_peaks(ecg_signal[rpeaks[i]:t_wave_end], height=[0, 0.5])
        if t_wave_peak.size == 0:
            continue
        t_wave_peak = rpeaks[i] + t_wave_peak[0]
        t_wave_end = rpeaks[i + 1]
        t_wave_duration = t_wave_end - t_wave_peak
        t_wave_durations.append(t_wave_duration/sampling_rate)

        # PR interval
        pr_interval = rpeaks[i] - p_wave_peak
        pr_intervals.append(pr_interval/sampling_rate)

        # QT interval
        qt_interval = t_wave_end - rpeaks[i]
        qt_intervals.append(qt_interval/sampling_rate)

    # Compute HRV metrics
    rr_intervals = np.diff(rpeaks)/sampling_rate
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

    # Calculate mean durations
    mean_p_wave_duration = np.mean(p_wave_durations)
    mean_qrs_duration = np.mean(qrs_durations)
    mean_t_wave_duration = np.mean(t_wave_durations)
    mean_qt_interval = np.mean(qt_intervals)
    mean_pr_interval = np.mean(pr_intervals)

    
    # Create DataFrame to store features
    ecg_features = pd.DataFrame({
        'SDNN': [sdnn],
        'RMSSD': [rmssd],
        'Mean_P_Wave_Duration': [mean_p_wave_duration],
        'Mean_QRS_Duration': [mean_qrs_duration],
        'Mean_T_Wave_Duration': [mean_t_wave_duration],
        'Mean_QT_Interval': [mean_qt_interval],
        'Mean_PR_Interval': [mean_pr_interval]
    })

    return ecg_features

# Define a function to normalize ECG signals
def normalize_ecg_signal(row):
    # Normalize ECG signal values
    normalized_signal = (row - row.min()) / (row.max() - row.min())
    
    return normalized_signal

# Define function to remove everything except letters and commas
def clean_string(string):
    return re.sub(r'[^a-zA-Z,]', '', string)

def extract_beats_df(ecg_df, window_size=10, threshold=0.9, padding_length=187):
    """
    Extracts beats from an ECG DataFrame.

    Parameters:
    ecg_df (pandas.DataFrame): DataFrame containing the ECG signals.
    window_size (int): Size of the window in seconds.
    threshold (float): Threshold for detecting R-peaks.
    padding_length (int): Length to pad each beat.

    Returns:
    pandas.DataFrame: DataFrame containing extracted beats.
    """
    # Extract the ECG signals as a 2D numpy array
    ecg_signals = ecg_df.iloc[:,0:1250].values

    # Calculate window length
    sampling_rate = 125  # Assuming ECG sampling rate of 500 Hz

    # Initialize an empty list to store beats
    beats_list = []

    # Iterate over each row index in the DataFrame
    for index in range(len(ecg_df)):
        if index % 100 == 0:
            print(index)
        # Extract beats from the current ECG signal
        row = ecg_signals[index]
        rpeaks = ecg.ecg(signal=row, sampling_rate=sampling_rate, show=False)['rpeaks']
        rr_intervals = np.diff(rpeaks)
        T = np.median(rr_intervals)
        if T > 187/1.2:
            T = 187//1.2
        row_beats = []
        for i in range(1, len(rpeaks)):
            start_idx = max(0, rpeaks[i - 1])
            end_idx = int(min(len(row), rpeaks[i - 1] + 1.2 * T))
            selected_part = row[start_idx:end_idx]
            padded_part = np.pad(selected_part, (0, padding_length - len(selected_part)), 'constant', constant_values=(0,))
            row_beats.append(padded_part)

        # Append beats to the beats_list along with other columns
        for beat in row_beats:
            beat_dict = {'Beat': beat}
            # Copy all other columns from the original row to the beat_dict
            for col_name, value in ecg_df.iloc[index].items():
                if not col_name.isdigit():
                    beat_dict[col_name] = value
            # Append the beat_dict to the beats_list
            beats_list.append(beat_dict)

    # Create a DataFrame from the beats_list
    beats_df = pd.DataFrame(beats_list)

    return beats_df

def apply_wavelet_transform(dataset):
    transformed_dataset = []
    for index, signal in dataset.iterrows():
        # Apply wavelet transform
        coeffs = pywt.wavedec(signal, 'db4', level=5)  # Adjust wavelet and level as needed
        # Flatten the coefficients and concatenate them
        flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])
        transformed_dataset.append(flattened_coeffs)
    
    # Create a DataFrame with column names coeff1, coeff2, etc.
    columns = [f'coeff{i}' for i in range(1, len(transformed_dataset[0]) + 1)]
    df = pd.DataFrame(transformed_dataset, columns=columns)
    return df

# Load raw dataset
X_train = pd.read_csv('X_train_PTB_XL_raw.csv')
X_test = pd.read_csv('X_test_PTB_XL_raw.csv')
Y_test = pd.read_csv('Y_test_PTB_XL_raw.csv')
Y_train = pd.read_csv('Y_train_PTB_XL_raw.csv') 

# Add features and labels to one dataframwe
Train_df = pd.concat([X_train, Y_train], axis=1)
Test_df = pd.concat([X_test, Y_test], axis=1)

# Remove rows with missing labels
Train_df = Train_df[Train_df['diagnostic_superclass'].apply(lambda x: x != '[]')]
Test_df = Test_df[Test_df['diagnostic_superclass'].apply(lambda x: x != '[]')]

# Split to features and labels
Y_test = Test_df['diagnostic_superclass']
Y_train = Train_df['diagnostic_superclass']
X_train = Train_df.drop(columns = 'diagnostic_superclass')
X_test = Test_df.drop(columns = 'diagnostic_superclass')

# Filter each signal
X_train_filtered_df = X_train.apply(filter_row, axis=1)
X_test_filtered_df = X_test.apply(filter_row, axis=1)

# Downsample each raw to 125Hz
X_train_downsampled_df = X_train_filtered_df.apply(downsample_row)
X_test_downsampled_df = X_test_filtered_df.apply(downsample_row)

# Change back to dataframe
X_train_downsampled_df = pd.DataFrame(X_train_downsampled_df.to_list(), columns=[f'{i}' for i in range(len(X_train_downsampled_df.iloc[0]))])
X_test_downsampled_df = pd.DataFrame(X_test_downsampled_df.to_list(), columns=[f'{i}' for i in range(len(X_test_downsampled_df.iloc[0]))])

# Extract feature for ECg signals
X_train_featrues_df = X_train_downsampled_df.apply(extract_ecg_features,axis = 1)
X_test_featrues_df = X_test_downsampled_df.apply(extract_ecg_features,axis = 1)

# Concatenate features with ECG signals
X_train_featrues_df_ = pd.concat(X_train_featrues_df.values, ignore_index = True)
X_test_featrues_df_ = pd.concat(X_test_featrues_df.values, ignore_index = True)

# Apply the normalize_ecg_signal function row-wise
X_train_normalized_df = X_train_downsampled_df.apply(normalize_ecg_signal, axis=1)
X_test_normalized_df = X_test_downsampled_df.apply(normalize_ecg_signal, axis=1)

# Combine normalized with features
X_train_combined = pd.concat([X_train_normalized_df, X_train_featrues_df_], axis = 1)
X_test_combined = pd.concat([X_test_normalized_df, X_test_featrues_df_], axis = 1)

# Change labels to strings
Y_train = Y_train.astype(str)
Y_test = Y_test.astype(str)

# Clean the strings from punctuation
Y_test_filtered = Y_test.apply(clean_string)
Y_train_filtered = Y_train.apply(clean_string)

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(Y_test_filtered.str.get_dummies(sep=','))

# Concatenate one-hot encoded features to original DataFrame
Y_test_encoded = pd.concat([Y_test_filtered, one_hot_encoded], axis=1)

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(Y_train_filtered.str.get_dummies(sep=','))

# Concatenate one-hot encoded features to original DataFrame
Y_train_encoded = pd.concat([Y_train_filtered, one_hot_encoded], axis=1)

Y_train_encoded['diagnostic_superclass'] = Y_train_encoded['diagnostic_superclass'].str.split(',')
Y_test_encoded['diagnostic_superclass'] = Y_test_encoded['diagnostic_superclass'].str.split(',')

# Reset index of labels
Y_test_encoded.index = range(len(Y_test_encoded))
Y_train_encoded.index = range(len(Y_train_encoded))

# Combine features and labels
Train_df_encoded_combined = pd.concat([X_train_combined,Y_train_encoded], axis =1)
Test_df_encoded_combined = pd.concat([X_test_combined, Y_test_encoded], axis = 1)

# Drop any rows with empty columns
Train_df_combined = Train_df_encoded_combined.dropna()
Test_df_combined = Test_df_encoded_combined.dropna()
Train_df_combined.index = range(len(Train_df_combined))
Test_df_combined.index = range(len(Test_df_combined))

# Extract beats for each signal
Test_beats_df = extract_beats_df(Test_df_combined)
Test_beats_expanded = pd.DataFrame(Test_beats_df['Beat'].to_list(), columns=[f'{i}' for i in range(len(Test_beats_df.iloc[0,0]))])
Test_beats_df = pd.concat([Test_beats_expanded, Test_beats_df.drop(columns='Beat')], axis = 1)
Train_beats_df = extract_beats_df(Train_df_combined)
Train_beats_expanded = pd.DataFrame(Train_beats_df['Beat'].to_list(), columns=[f'{i}' for i in range(len(Train_beats_df.iloc[0,0]))])
Train_beats_df = pd.concat([Train_beats_expanded, Train_beats_df.drop(columns='Beat')], axis = 1)

# Change name of dataframes
train_data = Train_beats_df
test_data = Test_beats_df

# Apply wavelet transform to train and test datasets
transformed_train_data = apply_wavelet_transform(train_data.iloc[:,:187])
transformed_test_data = apply_wavelet_transform(test_data.iloc[:,:187])

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Fit the scaler on the coefficients data and transform it
normalized_train_data = scaler.fit_transform(transformed_train_data)
normalized_test_data = scaler.transform(transformed_test_data)

# Convert the normalized arrays back to DataFrames
normalized_train_df = pd.DataFrame(normalized_train_data, columns=transformed_train_data.columns)
normalized_test_df = pd.DataFrame(normalized_test_data, columns=transformed_test_data.columns)

# Reset index
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Insert the transformed_train_data into the original DataFrame before the label column
train_data_with_coeffs = pd.concat([train_data.iloc[:, :187], normalized_train_df, train_data.iloc[:, 187:]], axis=1)

# Insert the transformed_test_data into the original DataFrame before the label column
test_data_with_coeffs = pd.concat([test_data.iloc[:, :187], normalized_test_df, test_data.iloc[:, 187:]], axis=1)
#%%

train_data_with_coeffs.to_csv('Train_PTB_XL.csv', index=False)
test_data_with_coeffs.to_csv('Test_PTB_XL.csv', index=False)