from preprocess.vocab import *
from public.path import path_data_dir
import numpy as np

from public import config

max_len = config.MAX_LEN


class PredictProcess(object):
    def __init__(self,
                 max_len = max_len,  # 'other'、'bert' bert 数据处理需要单独进行处理
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
    
    def get_data(self, txt):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """
        
        f = []
        for everyChar in txt:
            f.append(everyChar)
            if everyChar == '。' or everyChar == '；' or everyChar == '！' or everyChar == '？':
                f.append('\n')
        
        # 拼接地址
        # path_train = os.path.join(self.base_dir, "train.txt")
        # path_test = os.path.join(self.base_dir, "test.txt")
        # path_validate = os.path.join(self.base_dir, "validate.txt")
        
        # train_data = self.__bert_text_to_index(path_train)
        # test_data = self.__bert_text_to_index(path_test)
        validate_data = self.__text_to_indexs(f)
        # 进行 one-hot处理
        # if one_hot:
        #     def label_to_one_hot(index: []) -> []:
        #         data = []
        #         for line in index:
        #             data_line = []
        #             for i, index in enumerate(line):
        #                 line_line = [0] * self.tag_size
        #                 line_line[index] = 1
        #                 data_line.append(line_line)
        #             data.append(data_line)
        #         return np.array(data)
        
        # train_label = label_to_one_hot(index=train_label)
        # test_label = label_to_one_hot(index=test_label)
        # validate_label = label_to_one_hot(index=validate_label)
        # else:
        # train_label = np.expand_dims(train_label, 2)
        # test_label = np.expand_dims(test_label, 2)
        # validate_label = np.expand_dims(validate_label, 2)
        return validate_data
    
    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))
    
    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))
    
    def __text_to_indexs(self, f: []):
        data = []
        line_data = []
        for line in f:
            if line != '\n':
                # w, t = line.split()
                char_index = self.w2i.get(line, self.w2i[self.unk_flag])
                # tag_index = self.tag2index.get(t, 0)
                line_data.append(char_index)
                # line_label.append(tag_index)
            else:
                if len(line_data) < self.max_len:
                    pad_num = self.max_len - len(line_data)
                    line_data = [self.pad_index] * pad_num + line_data
                else:
                    line_data = line_data[:self.max_len]
                data.append(line_data)
                line_data = []
        
        return np.array(data)


if __name__ == '__main__':
    # dp = preprocess(data_type='data')
    # x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    #
    # print(y_train[:1, :1, :100])
    
    dp = PredictProcess()
    x_val = dp.get_data(
        "与2018年体检时胸部CT对比，结合MPR显示：纵膈各大血管结构清楚，血管间隙内未见明显肿大淋巴结。右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1cm*0.9cm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。双侧胸腔内未见明显胸水征。"
    )
    print(x_val)
    pass
