from predict.predict_bio import Model
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from public import config
from tqdm import tqdm

from public.path import test_file_path

max_len = config.MAX_LEN
verbose_each_sentense = False


class BIOLabel:
    def __init__(self, originLabel):
        self.originTxt = originLabel
        self.bioTag = self.originTxt.split('-')[0]
        self.label = self.originTxt.split('-')[1]


class ansListItem:
    def __init__(self, originDic):
        self.chars = originDic.get('chars')
        self.tags = originDic.get('tags')


def getBrat(ans_list):
    result = []
    i = 0
    startIndex = i
    endIndex = i
    word = ''
    while i < len(ans_list):
        # print(ans_list[i])
        if i == 0:
            if ans_list[i].get('tags') == 'O':
                pass
            elif BIOLabel(ans_list[i].get('tags')).bioTag == 'B':
                word += ansListItem(ans_list[i]).chars
                startIndex = i
                endIndex = i
        else:
            p = ansListItem(ans_list[i])
            pre = ansListItem(ans_list[i - 1])
            
            if p.tags == 'O':
                if pre.tags == 'O':
                    pass
                elif BIOLabel(pre.tags).bioTag == 'B':
                    result.append({
                        'word': word,
                        'start': startIndex,
                        'end': startIndex,
                        'label': BIOLabel(pre.tags).label
                    })
                    word = ''
                    startIndex = i
                    endIndex = startIndex
                elif BIOLabel(pre.tags).bioTag == 'I':
                    result.append({
                        'word': word,
                        'start': startIndex,
                        'end': endIndex,
                        'label': BIOLabel(pre.tags).label
                    })
                    word = ''
                    startIndex = i
                    endIndex = startIndex
            
            elif BIOLabel(p.tags).bioTag == 'B':
                if pre.tags == 'O':
                    startIndex = i
                    endIndex = i
                    word += p.chars
                elif BIOLabel(pre.tags).bioTag == 'B':
                    result.append({
                        'word': word,
                        'start': startIndex,
                        'end': startIndex,
                        'label': BIOLabel(pre.tags).label
                    })
                    word = ''
                    startIndex = i
                    endIndex = startIndex
                elif BIOLabel(pre.tags).bioTag == 'I':
                    result.append({
                        'word': word,
                        'start': startIndex,
                        'end': endIndex,
                        'label': BIOLabel(pre.tags).label
                    })
                    word = ''
                    startIndex = i
                    endIndex = startIndex
            
            elif BIOLabel(p.tags).bioTag == 'I':
                endIndex = i
                word += p.chars
        
        i += 1
    
    return result


# def evaluate(verbose_each_sentense = False):
sentenses = []
sample = []
tag_list = []
with open(test_file_path, 'r', encoding = 'utf-8') as f:
    sentense = ''
    tag = []
    for line in f:
        if line != '\n':
            word = line.split('\t')[0]
            sample.append({
                'chars': line.split('\t')[0],
                'tags': line.split('\t')[1].strip('\n')
            })
            sentense += word
        else:
            sentenses.append(sentense)
            sentense = ''

for i in range(len(sentenses)):
    if len(sentenses[i]) > max_len:
        sentenses[i] = sentenses[i][0:max_len]

# sentense = sentense.replace(',', '，').strip('"').strip().strip('\'')
bratListSample = getBrat(sample)
# print(sentense)

model = Model()
ans = []
ans_sentenses = []

if verbose_each_sentense:
    for i in range(0, len(sentenses)):
        ans_sentense = ''
        for j in model.predict(sentenses[i]):
            ans.append(j)
            ans_sentense += j['chars']
        ans_sentenses.append(ans_sentense)
        print(sentenses[i])
        print(ans_sentense)
        print('----------------------------')
else:
    for i in tqdm(range(0, len(sentenses))):
        ans_sentense = ''
        predicted = model.predict(sentenses[i])
        for j in predicted:
            ans.append(j)
            ans_sentense += j['chars']
        ans_sentenses.append(ans_sentense)

if verbose_each_sentense:
    print(len(sample))
    print(len(ans))

if verbose_each_sentense:
    for i in range(len(sample)):
        print(sample[i])
        print(ans[i])
        print('----------------------------')

bratListPredict = getBrat(ans)
if verbose_each_sentense:
    for i in bratListPredict:
        print(i)

a = 0
for i in bratListSample:
    for j in bratListPredict:
        if i['word'] == j['word'] and i['start'] == j['start'] and i['end'] == j['end'] \
                and i['label'] == j['label']:
            a += 1

print("TP:", a)
print("TP+FP:", len(bratListSample))
precision = a / len(bratListSample)
print("precision:", a / len(bratListSample))

b = 0
for i in bratListPredict:
    for j in bratListSample:
        if i['word'] == j['word'] and i['start'] == j['start'] and i['end'] == j['end'] \
                and i['label'] == j['label']:
            b += 1

print("TP+FN:", len(bratListPredict))
recall = b / len(bratListPredict)
print("recall:", b / len(bratListPredict))

print("f1:", (2 * precision * recall) / (precision + recall))

##
tag_ori = []
tag_pred = []
for i in range(len(sample)):
    tag_ori.append(sample[i]['tags'])
    tag_pred.append(ans[i]['tags'])

tag_ori = [tag_ori]
tag_pred = [tag_pred]
# print("accuary: ", accuracy_score(tag_ori, tag_pred))
# print("p: ", precision_score(tag_ori, tag_pred))
# print("r: ", recall_score(tag_ori, tag_pred))
# print("f1: ", f1_score(tag_ori, tag_pred))
print("\nclassification report: ")
print(classification_report(tag_ori, tag_pred))

# evaluate(verbose_each_sentense = False)
