import numpy as np
import pywt
import wfdb
import os

def denoise(data):
    # wavelet transform
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # Threshold denoising
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # Inverse wavelet transform to obtain the denoised signal
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def map_prediction_to_label(predicted_values):
    mapping = {0: 'N', 1: 'S', 2: 'V'}
    return np.array([mapping[val] for val in predicted_values])

# Hàm ánh xạ các ký hiệu trong Rclass thành lớp mới
def map_class_to_group(symbols):
    class_mapping = {
        'N': ['N', 'L', 'R', 'e', 'j'],
        'S': ['A', 'a', 'J', 'S'],
        'V': ['V', 'E']
    }
    mapped_classes = []
    for symbol in symbols:
        found = False
        for group, class_list in class_mapping.items():
            if symbol in class_list:
                mapped_classes.append(group)
                found = True
                break
        # Do nothing if not found
    return np.array(mapped_classes)

def getDataSet(number, X_data, Y_data, project_path):
    ecgClassSet = {
        'N': ['N', 'L', 'R', 'e', 'j'],
        'S': ['A', 'a', 'J', 'S'],
        'V': ['V', 'E']
    }
    
    _, info = wfdb.io.rdsamp(os.path.join(project_path, number))
    channels = info['sig_name']
    channel1 = channels[0]
    print(f"Channel: {channel1}")

    print(f"Reading ECG data for {number}...")
    record = wfdb.rdrecord(os.path.join(project_path, number), channel_names=[channel1])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    annotation = wfdb.rdann(os.path.join(project_path, number), 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    start = 2
    end = 3
    i = start
    j = len(Rclass) - end

    while i < j:
        try:
            beat_type = Rclass[i]
            label = None
            for class_label, beat_list in ecgClassSet.items():
                if beat_type in beat_list:
                    label = class_label
                    break

            if label is not None:
                if Rlocation[i] - 699 >= 0 and Rlocation[i] + 801 <= len(rdata):
                    x_train = rdata[Rlocation[i] - 699:Rlocation[i] + 801].astype(np.float32)
                    X_data.append(x_train)
                    Y_data.append(label)
            i += 1
        except ValueError:
            i += 1

    return X_data, Y_data

def get_record_raw(dataset):
    file = []
    for root, _, files in os.walk(dataset):
        for filename in files:
            if filename.endswith('.dat'):
                file.append(os.path.join(root, filename))
    return file

def prepare_datasets(project_path):
    all_files = get_record_raw(dataset=project_path)
    X_train, Y_train = [], []

    for number in all_files:
        number = number.split('\\')[-1].split('.')[0]
        print(f"Processing record: {number}")
        X_train, Y_train = getDataSet(number, X_train, Y_train, project_path)
        print(f"Processed {number}")

    X_train = np.array(X_train)
    Y_train = np.array([map_class_to_group(y) for y in Y_train])
    return X_train, Y_train
