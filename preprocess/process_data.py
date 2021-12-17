from preprocess.vocab import *
from public.path import path_data_dir
import numpy as np
import os


class DataProcess(object):
    def __init__(self,
                 max_len=100,
                 ):
        """
        数据处理
        :param max_len: 句子最长的长度，默认为保留100
        :param data_type: 数据类型，当前支持四种数据类型
        """
        self.w2i = get_w2i()  # word to index
        self.tag2index = get_tag2index()  # tag to index
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len

        self.unk_index = self.w2i.get(unk_flag, 2)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 3)
        self.sep_index = self.w2i.get(sep_flag, 4)

        self.base_dir = path_data_dir

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """
        # 拼接地址
        path_train = os.path.join(self.base_dir, "train.txt")
        path_test = os.path.join(self.base_dir, "test.txt")
        # path_validate = os.path.join(self.base_dir, "validate.txt")

        # 读取数据
        train_data, train_label = self.__text_to_indexs(path_train)
        test_data, test_label = self.__text_to_indexs(path_test)
        # validate_data, validate_label = self.__text_to_indexs(path_validate)

        # 进行 one-hot处理
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0] * self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)

            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
            # validate_label = label_to_one_hot(index=validate_label)
        else:
            train_label = np.expand_dims(train_label, 2)
            test_label = np.expand_dims(test_label, 2)
            # validate_label = np.expand_dims(validate_label, 2)
        return train_data, train_label, test_data, test_label  # , validate_data, validate_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    # texts 转化为 index序列
    def __text_to_indexs(self, file_path: str) -> ([], []):
        data, label = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            line_data, line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split('\t')
                    char_index = self.w2i.get(w, self.w2i[self.unk_flag])
                    tag_index = self.tag2index.get(t.strip('\n'), 0)
                    line_data.append(char_index)
                    line_label.append(tag_index)
                else:
                    if len(line_data) < self.max_len:
                        pad_num = self.max_len - len(line_data)
                        line_data = [self.pad_index] * pad_num + line_data
                        line_label = [0] * pad_num + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    label.append(line_label)
                    line_data, line_label = [], []
        return np.array(data), np.array(label)


if __name__ == '__main__':
    # dp = preprocess(data_type='data')
    # x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    #
    # print(y_train[:1, :1, :100])

    dp = DataProcess()
    x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(dp.get_data(one_hot=True)[0][0])
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(x_test[0].shape)
    print(x_test[1].shape)
    print(y_test.shape)

    print(y_train[:1, :1, :100])
