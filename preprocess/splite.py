if __name__ == '__main__':
    file = open('../data/data2/k.txt')
    data = []
    for line in file.readlines():
        curLine = line.strip().split("\n")
        data.append(curLine[0])

    train = []
    validate = []
    test = []

    sentense = []

    one_sentense = []
    count = 0
    for i in range(0, len(data)):
        one_sentense.append(data[i])
        if data[i] == '':
            count += 1
            sentense.append(one_sentense)
            one_sentense = []

    print(count)
    print(sentense)
    total = len(sentense)

    train_num = 0
    val_num = 0
    test_num = 0

    for i in range(0, len(sentense)):
        if i / len(sentense) <= 0.6:
            train_num += 1
            for one in sentense[i]:
                train.append(one)
        elif i / len(sentense) < 0.8:
            val_num += 1
            for one in sentense[i]:
                validate.append(one)
        else:
            test_num += 1
            for one in sentense[i]:
                test.append(one)


    print(train_num)
    print(val_num)
    print(test_num)
    print()
    f1 = open("../data/data/train.txt", "w")

    for line in train:
        f1.write(line + '\n')
    f1.close()

    f2 = open("../data/data/test.txt", "w")

    for line in test:
        f2.write(line + '\n')
    f2.close()
