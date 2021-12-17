import os

from public.path import weight_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.BiLSTMCRF import BILSTMCRF

from sklearn.metrics import f1_score, recall_score
import numpy as np
import pandas as pd

from public.utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from preprocess.process_data import DataProcess
from public.config import *

max_len = MAX_LEN


def train_sample(epochs = 15, log = None):
    dp = DataProcess(max_len = max_len)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot = True)
    print('train_data:')
    print(train_data[0])
    
    log.info("----------------------------数据信息 START--------------------------")
    log.info(f"当前使用数据集 MSRA")
    log.info(f"train_label:{train_label.shape}")
    log.info(f"test_label:{test_label.shape}")
    log.info("----------------------------数据信息 END--------------------------")
    
    model_class = BILSTMCRF(dp.vocab_size, dp.tag_size)
    
    model = model_class.creat_model()
    
    callback = TrainHistory(log = log, model_name = "BILSTMCRF")  # 自定义回调 记录训练数据
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 1)
    early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)  # 提前结束
    save_model = ModelCheckpoint(weight_path, monitor = 'loss', verbose = 1, save_best_only = True,
                                 mode = 'min')
    
    model.fit(train_data, train_label, batch_size = BATCH_SIZE, epochs = epochs,
              callbacks = [callback, early_stopping, reduce_lr, save_model])
    
    result = model.evaluate(test_data, test_label)
    print("Test Set: [Loss , Accuracy]:")
    print(result)
    # 计算 f1 和 recall值
    
    pre = model.predict(test_data)
    pre = np.array(pre)
    test_label = np.array(test_label)
    pre = np.argmax(pre, axis = 2)
    test_label = np.argmax(test_label, axis = 2)
    pre = pre.reshape(pre.shape[0] * pre.shape[1], )
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], )
    
    f1score = f1_score(pre, test_label, average = 'weighted')
    recall = recall_score(pre, test_label, average = 'weighted')
    
    log.info("================================================")
    log.info(f"--------------:f1: {f1score} --------------")
    log.info(f"--------------:recall: {recall} --------------")
    log.info("================================================")
    
    # 把 f1 和 recall 添加到最后一个记录数据里面
    info_list = callback.info
    if info_list and len(info_list) > 0:
        last_info = info_list[-1]
        last_info['f1'] = f1score
        last_info['recall'] = recall
        last_info['test_loss'] = result[0]
        last_info['test_val'] = result[1]
    
    return info_list


if __name__ == '__main__':
    # 定义文件路径（以便记录数据）
    log_path = os.path.join(path_log_dir, 'train_log.log')
    df_path = os.path.join(path_log_dir, 'df.csv')
    log = create_log(log_path)
    
    # 训练同时记录数据写入的df文件
    columns = ['model_name', 'epoch', 'loss', 'acc', 'f1', 'recall', 'test_loss', 'test_val']
    df = pd.DataFrame(columns = columns)
    info_list = train_sample(epochs = EPOCHS, log = log)
    for info in info_list:
        df = df.append([info])
    df.to_csv(df_path)
