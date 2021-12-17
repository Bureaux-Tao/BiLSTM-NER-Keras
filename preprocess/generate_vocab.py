from collections import Counter

from public.path import train_file_path, test_file_path, path_vocab

lines = []
with open(train_file_path, encoding = 'utf-8') as f1:
    for i in f1:
        if i.strip('\n') != '':
            lines.append(i.strip('\n').split('\t')[0])

with open(test_file_path, encoding = 'utf-8') as f2:
    for j in f2:
        if j.strip('\n') != '':
            lines.append(j.strip('\n').split('\t')[0])

print(len(lines))
freq = dict(Counter(lines))
print(sorted(freq.items(), key = lambda d: d[1], reverse = True))
freq_list = sorted(freq.items(), key = lambda d: d[1], reverse = True)

with open(path_vocab, "a", encoding = 'utf-8') as f3:
    f3.write("{}\n".format(str('[PAD]')))
    f3.write("{}\n".format(str('[UNK]')))
    f3.write("{}\n".format(str('[CLS]')))
    f3.write("{}\n".format(str('[SEP]')))
    f3.write("{}\n".format(str('[MASK]')))
    for i in freq_list:
        f3.write("{}\n".format(str(i[0])))
