from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Masking, Embedding, Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation,Dropout,CuDNNLSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy,crf_accuracy
from tqdm import trange
import numpy as np
import pandas as pd
import csv
import re
from keras_bert import extract_embeddings
model_path = '/home/zengjiaqi/chinese_L-12_H-768_A-12'

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

import math


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)


max_length=100
zero = [0 for i in range(768)]#零向量 768维
def create_train_text(input_file):
    print("begin create embedding")
    csv_read = pd.read_csv(input_file,encoding='gbk')
    embedding_pad = []
    texts = []
    for i in trange (10000):#语句数目根据需要更改
        string=str(csv_read.loc[i][3])
        for number in re.findall(r"\d",string):#将数字用*代替
            string=string.replace(number,'*')
        texts.append(string)
    print("begin extract embeddings")
    embedding = extract_embeddings(model_path, texts, batch_size=64) #keras_bert中已包装好的用bert提取embedding的函数
    for i in trange(len(embedding)):
        e=embedding[i][1:len(embedding[i])-1].tolist()
        for j in range(max_length-len(e)):#补零
            e.append(zero)
        embedding_pad.append(e)

    input = np.asarray(embedding_pad)
    print(input.shape)
    np.save('financialnews10000*',input)
    return input

tag2id = {"B":1,"I":2,"O":0}

def create_train_label(input_file): #根据标注的csv中的格式生成label，用120表示BIO
    csv_read = pd.read_csv(input_file,encoding='gbk')
    label=[]
    for i in trange(1000):
        labeli=[]
        if (int(csv_read.loc[i][5])==0):#不包含事件
            for j in range(int(csv_read.loc[i][4])):
                labeli.append(0)
        else:
            for j in range(int(csv_read.loc[i][5])-1):
                labeli.append(0)#事件前面的O
            labeli.append(1)#事件开始
            for j in range(int(csv_read.loc[i][5]),int(csv_read.loc[i][6])):
                labeli.append(2)#事件继续
            if (str(csv_read.loc[i][7])=='nan'):#句子中只有一个事件
                for j in range(int(csv_read.loc[i][6]),int(csv_read.loc[i][4])):
                    labeli.append(0)#事件后面的O
            else:#不止一个事件

                    for j in range(int(csv_read.loc[i][6]),int(csv_read.loc[i][7])-1):
                        labeli.append(0)#两个事件中间的O
                    labeli.append(1)#第二个事件开始
                    for j in range(int(csv_read.loc[i][7]),int(csv_read.loc[i][8])):
                        labeli.append(2)#第二个事件继续
                    if (str(csv_read.loc[i][9])=='nan'):#句子中只有两个事件
                        for j in range(int(csv_read.loc[i][8]),int(csv_read.loc[i][4])):
                            labeli.append(0)#第二个事件结束后的0
                    else:#有三个事件
                        for j in range(int(csv_read.loc[i][8]),int(csv_read.loc[i][9])-1):
                            labeli.append(0)#第二个和第三个事件中间的0
                        labeli.append(1)#第三个事件开始
                        for j in range(int(csv_read.loc[i][9]),int(csv_read.loc[i][10])):
                            labeli.append(2)#第三个事件继续
                        #最多只有三个事件
                        for j in range(int(csv_read.loc[i][10]),int(csv_read.loc[i][4])):
                            labeli.append(0)#第三个事件结束后的0
        for j in range(len(labeli),max_length):
            labeli.append(0)#补足长度
        label.append(labeli)
    #print(label)
    label_array = np.asarray(label)
    np.save('y_train',label_array)
    #print(label_array.shape)
    return label_array


def nerual_network():
    #neural network
    HIDDEN_UNITS=128
    NUM_CLASS=3
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(HIDDEN_UNITS, return_sequences=True),input_shape=(100,768)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(CuDNNLSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf_layer = CRF(NUM_CLASS,sparse_target=False,learn_mode = 'marginal')
    model.add(crf_layer)
    model.summary()
    model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
    return model


	#history = model.fit(input_train,output_train,validation_split=0.15,batch_size=500,epochs=500,callbacks=[early_stopping])

'''
input_file = 'financialnews.com.cn_extract_for_event.csv'
x_train = create_train_text(input_file)
y_train = create_train_label(input_file)
y_train = to_categorical(y_train)
print(y_train,y_train.shape)
'''

def train(x_train,y_train):
    print(x_train.shape,y_train.shape)
    model = nerual_network()
    early_stopping = EarlyStopping(monitor='val_loss',patience=20,verbose=1,mode='auto')
    model.fit(x_train,y_train,epochs=10,batch_size=16,validation_split=0.2)#,callbacks=[early_stopping])
    model.save_weights('postag_weights*_prob.h5')

def test(x_test,test_file):
    model=nerual_network()
    model.load_weights('postag_weights*_prob.h5')
    y_test = model.predict(x_test)
    #print(y_test[0])
    '''求联合置信概率'''
    prob=[1 for i in range(y_test.shape[0])]
    for j in range(y_test.shape[0]):
        for i in range(y_test[j].shape[0]):
            prob[j]=prob[j]*np.max(y_test[j][i])
    print('prob:',prob)
    count=0
    for i in range(len(prob)):
        if (prob[i]<0.8): #计算置信概率小于0.8的语句
            count=count+1
    print('count:',count,count/len(prob))


    print(y_test.shape)
    output=[]
    for i in range(y_test.shape[0]):
        output_i=[]
        for j in range(y_test.shape[1]):
            output_i.append(np.argmax(y_test[i][j]))
        output.append(output_i)
    output=np.asarray(output)
    print(output.shape) #output为每句话的[0,0,1,2,2,2,2,0,0,……]
    #print(output[0:100])
    result = [] #result为每句话中事件的起止位置如[2,6]
    for i in range(output.shape[0]):
        result_i=[]
        for j in range(output.shape[1]-1):
            if (j==0 and output[i][j]!=0):
                result_i.append(j) #对于开头为1或2
            if (output[i][j]==0 and output[i][j+1]!=0):
                result_i.append(j+1)
            if (output[i][j]!=0 and output[i][j+1]==0):
                result_i.append(j)
            if (j==output.shape[1]-2 and output[i][j+1]!=0):
                result_i.append(j+1) #对于结尾为……012
        result.append(result_i)
    #print(result)
    '''
    for i in range(len(result)): #对应人工标注 从1开始
        for j in range(len(result[i])):
            result[i][j]=result[i][j]+1
    '''
    print(result)
    test_csv = pd.read_csv(test_file,encoding='gbk')
    f=open('result_9000_prob.csv','a+')
    csv_write = csv.writer(f)
    for i in range(len(result)):
        if(len(result[i])==0):
            csv_write.writerow([test_csv.loc[i+1000][3]])
        else:
            row=[]
            row.append(test_csv.loc[i+1000][3])
            for j in range(0,len(result[i]),2):
                row.append(result[i][j]+1)#+1对应人手工标注
                row.append(result[i][j+1]+1)
            for j in range(0,len(result[i]),2):
                row.append(test_csv.loc[i+1000][3][result[i][j]:result[i][j+1]+1])#文字内容
            csv_write.writerow(row)
    f.close()


def main():
    input_file = 'financialnews.com.cn_extract_for_event.csv'
    #test_file = 'qihuowang_extract_for_event.csv'
    test_file = 'financialnews.com.cn_extract_for_event.csv'
    #x_train = np.load('x_train*.npy')
    #y_train = np.load('y_train.npy')
    #y_train = to_categorical(y_train)
    #train(x_train,y_train)
    x_test=np.load('financialnews10000*.npy',mmap_mode='r')

    #x_test = np.load('x_test_qihuo1000.npy')
    #print(x_test.shape)
    test(x_test[1000:],test_file)

main()
test_file = 'qihuowang_extract_for_event.csv'
input_file = 'financialnews.com.cn_extract_for_event.csv'
#create_train_text(input_file)
