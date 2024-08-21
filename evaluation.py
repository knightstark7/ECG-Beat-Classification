import wfdb 
import os
import numpy as np
from collections import Counter
from data_processing import denoise, map_class_to_group, map_prediction_to_label
from model import build_model
from config import TEMP_DIR, MITDB_DIR, project_path, RESULT_DIR
import tensorflow as tf
from data_processing import get_record_raw, getDataSet
from multiprocessing import Pool
from ec57 import ec57_eval

def evaluate(file, predicted, dataset, ec57=True, test_data=[]):
    
    # Đọc dữ liệu và thông tin về kênh
    _, info = wfdb.io.rdsamp(os.path.join(dataset, file))
    channels = info['sig_name']
    channel1, channel2 = channels[0], channels[1]
    
    # Đọc dữ liệu ECG
    print("Reading " + file + " ECG data...")
    record = wfdb.rdrecord(os.path.join(dataset, file), channel_names=[channel1])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    signal_length = len(rdata)
    
    # Đọc thông tin annotation
    annotation = wfdb.rdann(os.path.join(dataset, file), 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    
    # Xác định điều kiện để chọn các ký hiệu hợp lệ
    valid_symbols = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
    valid_condition = np.isin(Rclass, valid_symbols)

    # Lọc các ký hiệu và vị trí hợp lệ
    sample = np.array(Rlocation)[valid_condition]
    Rclass = np.array(Rclass)[valid_condition]
    
    # Định nghĩa start và end indices cho việc labeling
    start = 2
    end = 3
    i = start
    j = len(Rclass) - end

    # Lọc sample và Rclass dựa trên phạm vi hợp lệ
    valid_indices = []
    while i < j:
        if sample[i] - 699 >= 0 and sample[i] + 801 <= signal_length:
            valid_indices.append(i)
        i += 1

    sample = sample[valid_indices]
    Rclass = Rclass[valid_indices]
    
    # Ánh xạ ký hiệu Rclass thành lớp nhóm
    grouped_Rclass = map_class_to_group(Rclass)
    

    # Ánh xạ giá trị dự đoán thành nhãn
    predicted_labels = map_prediction_to_label(predicted.astype(int))
    
    # Đảm bảo valid_indices nằm trong phạm vi của predicted_labels
    valid_indices = [idx for idx in valid_indices if idx < len(predicted_labels)]

    # Đồng bộ hóa số lượng beat
    sample = sample[:len(valid_indices)]
    grouped_Rclass = grouped_Rclass[:len(valid_indices)]
    predicted_labels = predicted_labels[:len(valid_indices)]

    # Chuyển đổi dữ liệu thành numpy.ndarray nếu cần
    sample = np.array(sample)
    grouped_Rclass = np.array(grouped_Rclass)
    predicted_labels = np.array(predicted_labels)

    # Ghi beat và sample vào các tệp
    if ec57:
        ann_dir = os.path.join(TEMP_DIR, dataset.split('/')[-2])
        print(ann_dir)
        # Ghi các beat và nhãn gốc
        wfdb.wrann(file, extension='atr', sample=sample, symbol=grouped_Rclass, write_dir=ann_dir)
        # Ghi các beat và dự đoán
        wfdb.wrann(file, extension='pred', sample=sample, symbol=predicted_labels, write_dir=ann_dir)
        print('Done')
        return

# Hyperparameters
max_sequence_length = 1500
num_channels = 1
d_model = 128
num_heads = 4
dropout_rate = 0.4
dff = 128

def multi_predict(file_number, dataset, flag, num):
    X_data = []
    y_data = []
    test_data, y_data = getDataSet(file_number, X_data, y_data, project_path)
    test_indices = test_data
    test_data = np.array(test_data)
    test_data = test_data.reshape(-1, 1500, 1)
    
    with tf.device("/GPU:0"):
        model = build_model(max_sequence_length, num_channels, d_model, num_heads, dff, dropout_rate)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_path = project_path + "trans1_smote_tomek_39.keras"
        model.load_weights(model_path)
        prediction = np.argmax(model.predict(test_data), axis=-1)
        print('Pre:', prediction.shape)
    prediction = np.rint(prediction)
    evaluate(file_number, prediction, dataset, flag, test_indices)

def get_result_ec57():
    database = [
        MITDB_DIR,
        # AHADB_DIR,
        # ESCDB_DIR,
        # NSTDB_DIR
    ]

    #mitdb files
    valid_files = ['100', '101', '103', '105', '106', '108', '109',
                '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', 
                '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', 
                '209', '210', '212', '213', '214', '215', '219', '220', '221', 
                '222', '223', '228', '230', '231', '232', '233', '234']
    #nstdb files
    # valid_files = ['118e00', '118e06', '118e12', '118e18', '118e24', '118e_6', '119e00', '119e06', '119e12', '119e18', '119e24', '119e_6']
    for dataset in database:
        ann_dir = TEMP_DIR + MITDB_DIR
        # print(ann_dir)
        if not os.path.isdir(ann_dir):
            os.chmod(ann_dir, 0o666)
            os.makedirs(ann_dir)
        arg_list = []
        for file in get_record_raw(dataset):
            file_number = file.split('/')[-1][:-4]
            if file_number not in valid_files:
                continue
            print(file_number)
            arg_list.append([file_number, dataset, True, 3])

        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(multi_predict, arg_list)

        result_dir = RESULT_DIR
        if not os.path.isdir(result_dir):
            os.chmod(result_dir, 0o666)
            os.makedirs(result_dir)
        ec57_eval(result_dir, ann_dir, 'atr', 'atr', 'pred', None)
        
        