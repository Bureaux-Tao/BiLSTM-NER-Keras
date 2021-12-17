from public.path import test_file_path

if __name__ == '__main__':
    file = open(test_file_path)
    data = []
    for line in file.readlines():
        curLine = line.strip().split("\n")
        data.append(curLine[0])

    gen = []

    count1=0
    count2=0
    count3=0
    for i in range(1, len(data) - 1):
        gen.append(data[i])
        # print(type(data[i]))
        if data[i] == "。 O" and data[i + 1] != '" O':
            count1+=1
            gen.append("\n")
        if data[i] == '" O' and data[i - 1] == "。 O":
            gen.append('\n')
            count2+=1
        if data[i] == '； O' or data[i] == '？ O':
            gen.append('\n')
            count3+=1


    for j in gen:
        print(j)

    print(count1)
    print(count2)
    print(count3)
    print(count1+count2+count3)

    f = open("../data/data2/k.txt", "w")

    for line in gen:
        f.write(line + '\n')
    f.close()
