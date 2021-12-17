import sys

import numpy as np

from predict import predict_process
from public.config import BATCH_SIZE
from keras_contrib.layers import CRF

from preprocess.vocab import get_word, get_label
from model.BiLSTMCRF import BILSTMCRF
from public import config
from public.path import weight_path

max_len = config.MAX_LEN


class Model:
    def __init__(self):
        
        self.dp = predict_process.PredictProcess()
        
        self.model_class = BILSTMCRF(self.dp.vocab_size, self.dp.tag_size)
        
        self.model = self.model_class.creat_model()
        self.model.load_weights(weight_path)
    
    def __prepare(self, txt):
        validate_data = self.dp.get_data(txt)
        pre = self.model.predict(validate_data, batch_size = BATCH_SIZE)
        
        return [validate_data, pre]
    
    def predict(self, s):
        
        pre = self.__prepare(
            # "与2018年体检时胸部CT对比，结合MPR显示：纵膈各大血管结构清楚，血管间隙内未见明显肿大淋巴结。右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1cm*0.9cm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。双侧胸腔内未见明显胸水征。"
            s
        )
        ori = pre[0]
        
        # print(pre[1])
        label = np.argmax(pre[1], axis = 2)
        # print('------------------label---------------------')
        
        ans_list = []
        for (i_word, i_tag) in zip(ori, label):
            for (j_word, j_tag) in zip(i_word, i_tag):
                if j_word != 1 and j_word != 3 and j_word != 4:
                    # print(get_word(j_word), get_label(j_tag))
                    ans_list.append({'chars': get_word(j_word), 'tags': get_label(j_tag)})
        
        # 把不符合BIO规则的处理了
        for i in range(1, len(ans_list) - 1):
            p = ans_list[i]
            pre = ans_list[i - 1]
            if p['tags'] != 'O':
                if p['tags'].split('-')[0] == 'I':
                    if pre['tags'] == 'O':
                        p['tags'] = 'B-' + p['tags'].split('-')[1]
                    elif pre['tags'] != p['tags'] and pre['tags'].split('-')[0] != 'B':
                        p['tags'] = 'B-' + p['tags'].split('-')[1]
                    elif pre['tags'] != p['tags'] and pre['tags'].split('-')[1] != p['tags'].split('-')[1]:
                        p['tags'] = 'B-' + p['tags'].split('-')[1]
        
        # for i in ans_list:
        #     tag = i['tags'].split('-')
        #     if len(tag) == 2:
        #         i['tags'] = tag[1] + '-' + tag[0]
        #         print(i['chars'], i['tags'])
        
        return ans_list


if __name__ == '__main__':
    lstm = Model()
    txt = """
右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1*0.9mm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。
"""
    r = lstm.predict(txt)
    for i in r:
        print(i)
