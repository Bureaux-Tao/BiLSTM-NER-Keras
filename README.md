# Chinese named entity recognization (Bilstm with Keras)

## Project Structure

```
./
├── README.md
├── data
│   ├── README.md
│   ├── data							数据集
│   │   ├── test.txt
│   │   └── train.txt
│   ├── plain_text.txt
│   └── vocab.txt                       词表
├── evaluate
│   ├── __init__.py
│   └── f1_score.py                     计算实体F1得分
├── keras_contrib                       keras_contrib包，也可以pip装
├── log                                 训练nohup日志
│   ├── __init__.py
│   └── nohup.out
├── model                               模型
│   ├── BiLSTMCRF.py
│   ├── __init__.py
│   └── __pycache__
├── predict                             输出预测
│   ├── __init__.py
│   ├── __pycache__
│   ├── predict.py
│   └── predict_process.py
├── preprocess                          数据预处理
│   ├── README.md
│   ├── __pycache__
│   ├── convert_jsonl.py
│   ├── data_add_line.py
│   ├── generate_vocab.py               生成词表
│   ├── process_data.py                 数据处理转换
│   ├── splite.py
│   └── vocab.py                        词表对应工具
├── public
│   ├── __init__.py
│   ├── __pycache__
│   ├── config.py                       训练设置
│   ├── generate_label_id.py            生成label2id文件
│   ├── label2id.json                   标签dict
│   ├── path.py                         所有路径
│   └── utils.py                        小工具
├── report
│   └── report.out                      F1评估报告
├── train.py
└── weight                              保存的权重
    └── bilstm_ner.h5

52 directories, 214 files
```

## Dataset

三甲医院肺结节数据集，20000+字，BIO格式，形如：

```
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```
ATTENTION: 在处理自己数据集的时候需要注意：
- 字与标签之间用tab（"\t"）隔开
- 其中句子与句子之间使用空行隔开

## Steps

1. 替换数据集
2. 修改public/path.py中的地址
3. 使用public/generate_label_id.py生成label2id.txt文件，将其中的内容填到preprocess/vocab.py的get_tag2index中。注意：序号必须从0开始
4. 修改public/config.py中的MAX_LEN（超过截断，少于填充，最好设置训练集、测试集中最长句子作为MAX_LEN）
5. 运行preprocess/generate_vocab.py生成词表，词表按词频生成
6. 根据需要修改BiLSTMCRF.py模型结构
7. 修改public/config.py的参数
8. 训练前**debug看下train_data,train_label对不对**
9. 训练

## Model

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, None, 128)         81408
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 256)         263168
_________________________________________________________________
dropout_1 (Dropout)          (None, None, 256)         0
_________________________________________________________________
bidirectional_2 (Bidirection (None, None, 128)         164352
_________________________________________________________________
dropout_2 (Dropout)          (None, None, 128)         0
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 29)          3741
_________________________________________________________________
dropout_3 (Dropout)          (None, None, 29)          0
_________________________________________________________________
crf_1 (CRF)                  (None, None, 29)          1769
=================================================================
Total params: 514,438
Trainable params: 514,438
Non-trainable params: 0
_________________________________________________________________
```

## Train

运行train.py

```
Epoch 1/500
806/806 [==============================] - 15s 18ms/step - loss: 2.4178 - crf_viterbi_accuracy: 0.9106

Epoch 00001: loss improved from inf to 2.41777, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 2/500
806/806 [==============================] - 10s 13ms/step - loss: 0.6370 - crf_viterbi_accuracy: 0.9106

Epoch 00002: loss improved from 2.41777 to 0.63703, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 3/500
806/806 [==============================] - 11s 14ms/step - loss: 0.5295 - crf_viterbi_accuracy: 0.9106

Epoch 00003: loss improved from 0.63703 to 0.52950, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 4/500
806/806 [==============================] - 11s 13ms/step - loss: 0.4184 - crf_viterbi_accuracy: 0.9064

Epoch 00004: loss improved from 0.52950 to 0.41838, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 5/500
806/806 [==============================] - 12s 14ms/step - loss: 0.3422 - crf_viterbi_accuracy: 0.9104

Epoch 00005: loss improved from 0.41838 to 0.34217, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 6/500
806/806 [==============================] - 10s 13ms/step - loss: 0.3164 - crf_viterbi_accuracy: 0.9106

Epoch 00006: loss improved from 0.34217 to 0.31637, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 7/500
806/806 [==============================] - 10s 12ms/step - loss: 0.3003 - crf_viterbi_accuracy: 0.9111

Epoch 00007: loss improved from 0.31637 to 0.30032, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 8/500
806/806 [==============================] - 10s 12ms/step - loss: 0.2906 - crf_viterbi_accuracy: 0.9117

Epoch 00008: loss improved from 0.30032 to 0.29058, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 9/500
806/806 [==============================] - 9s 12ms/step - loss: 0.2837 - crf_viterbi_accuracy: 0.9118

Epoch 00009: loss improved from 0.29058 to 0.28366, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 10/500
806/806 [==============================] - 9s 11ms/step - loss: 0.2770 - crf_viterbi_accuracy: 0.9142

Epoch 00010: loss improved from 0.28366 to 0.27696, saving model to /home/bureaux/Projects/BiLSTMCRF_TimeDistribute/weight/bilstm_ner.h5
Epoch 11/500
806/806 [==============================] - 10s 12ms/step - loss: 0.2713 - crf_viterbi_accuracy: 0.9160
```

## Evaluate

运行evaluate/f1_score.py

```
100%|█████████████████████████████████████████| 118/118 [00:38<00:00,  3.06it/s]
TP: 441
TP+FP: 621
precision: 0.7101449275362319
TP+FN: 604
recall: 0.7301324503311258
f1: 0.72

classification report:
              precision    recall  f1-score   support

     ANATOMY       0.74      0.75      0.74       220
    BOUNDARY       1.00      0.75      0.86         8
     DENSITY       0.78      0.88      0.82         8
    DIAMETER       0.82      0.88      0.85        16
     DISEASE       0.54      0.72      0.62        43
   LUNGFIELD       0.83      0.83      0.83         6
      MARGIN       0.57      0.67      0.62         6
      NATURE       0.00      0.00      0.00         6
       ORGAN       0.62      0.62      0.62        13
    QUANTITY       0.88      0.87      0.87        83
       SHAPE       1.00      0.43      0.60         7
        SIGN       0.66      0.65      0.65       189
     TEXTURE       0.75      0.43      0.55         7
   TREATMENT       0.25      0.33      0.29         9

   micro avg       0.71      0.71      0.71       621
   macro avg       0.67      0.63      0.64       621
weighted avg       0.71      0.71      0.71       621
```

## Predict

运行predict/predict_bio.py