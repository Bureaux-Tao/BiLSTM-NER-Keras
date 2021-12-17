import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = os.path.dirname(current_dir)

# 词表目录(本词表采用的是 bert预训练模型的词表)
path_vocab = proj_path + '/data/vocab.txt'

# 实体命名识别文件目录
path_data_dir = proj_path + '/data/data/'
path_msra_dir = proj_path + '/data/MSRA/'
path_renmin_dir = proj_path + '/data/renMinRiBao/'

# bert 预训练文件地址
path_bert_dir = proj_path + '/data/chinese_L-12_H-768_A-12/'
# 日志、记录类文件目录地址
path_log_dir = proj_path + "/log"

train_file_path = proj_path + "/data/data/train.txt"
test_file_path = proj_path + "/data/data/test.txt"

weight_path = proj_path + "/weight/bilstm_ner.h5"
label2id_path = proj_path + "/public/label2id"

if __name__ == '__main__':
    print(path_bert_dir)
