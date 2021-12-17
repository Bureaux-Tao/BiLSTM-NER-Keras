# 获取词典

from public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r', encoding = 'utf-8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            # print(text)
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


def get_word(value):
    return [k for k, v in get_w2i().items() if v == value][0]


def get_label(value):
    return [k for k, v in get_tag2index().items() if v == int(value)][0]


# 获取 tag to index 词典
# def get_tag2index():
#     return {"O": 0,
#             "B-PER": 1, "I-PER": 2,
#             "B-LOC": 3, "I-LOC": 4,
#             "B-ORG": 5, "I-ORG": 6
#             }

def get_tag2index():
    return {
        "O": 0,
        "B-ANATOMY": 1,
        "I-ANATOMY": 2,
        "B-SIGN": 3,
        "I-SIGN": 4,
        "B-QUANTITY": 5,
        "I-QUANTITY": 6,
        "B-ORGAN": 7,
        "I-ORGAN": 8,
        "B-TEXTURE": 9,
        "I-TEXTURE": 10,
        "B-DISEASE": 11,
        "I-DISEASE": 12,
        "B-DENSITY": 13,
        "I-DENSITY": 14,
        "B-BOUNDARY": 15,
        "I-BOUNDARY": 16,
        "B-MARGIN": 17,
        "I-MARGIN": 18,
        "B-DIAMETER": 19,
        "I-DIAMETER": 20,
        "B-SHAPE": 21,
        "I-SHAPE": 22,
        "B-TREATMENT": 23,
        "I-TREATMENT": 24,
        "B-LUNGFIELD": 25,
        "I-LUNGFIELD": 26,
        "B-NATURE": 27,
        "I-NATURE": 28
    }


if __name__ == '__main__':
    while True:
        flt = input()
        print(get_word(float(flt)))
    # print(get_w2i().keys())
