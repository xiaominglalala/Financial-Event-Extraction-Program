
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from tqdm import trange
from torchcrf import CRF
import pandas as pd
import csv
import numpy as np
import re
from keras_bert import extract_embeddings
model_path = '/home/zengjiaqi/chinese_L-12_H-768_A-12'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
sequence_length = 100
input_size = 768
hidden_size = 128
num_layers = 3
num_classes = 3
batch_size = 16
num_epochs = 10
learning_rate = 0.001
max_length=100
zero = [0 for i in range(768)]#零向量 768维

def construct_dataset():
    x_train = np.load('x_train.npy',mmap_mode='r')
    y_train = np.load('y_train.npy',mmap_mode='r')
    x_valid = np.load('x_valid.npy',mmap_mode='r')
    y_valid = np.load('y_valid.npy',mmap_mode='r')
    x_test = np.load('x_test.npy',mmap_mode='r')
    y_test = np.load('y_test.npy',mmap_mode='r')
    print(x_train.shape,x_valid.shape,x_test.shape)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train)
    x_valid = torch.from_numpy(x_valid).float()
    y_valid = torch.from_numpy(y_valid)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test)

    train_dataset = TensorDataset(x_train,y_train)
    validation_dataset = TensorDataset(x_valid,y_valid)
    test_dataset = TensorDataset(x_test,y_test)#临时凑数

    #Data Loader
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size = batch_size, shuffle=False)
    return train_loader,validation_loader,test_loader

def create_embedding_x(input_file,range1,range2):
    print("begin create embedding")

    csv_read = pd.read_csv(input_file,encoding='utf8')#或者gbk
    embedding_pad = []
    texts = []
    for i in trange (range1,range2):#语句数目根据需要更改
        string=str(csv_read.loc[i][4])
        #print(string,len(string))
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

    input = np.asarray(embedding_pad).astype('float32')
    print(input.shape)
    np.save('x_train',input)
    return input

def create_label_y(input_file,range1,range2): #根据标注的csv中的格式生成label，用120表示BIO
    csv_read = pd.read_csv(input_file,encoding='utf8')
    label=[]
    for i in trange(range1,range2):
        labeli=[]
        #print(csv_read.loc[i][3])
        if (int(csv_read.loc[i][6])==0):#不包含事件
            for j in range(int(csv_read.loc[i][5])):
                labeli.append(0)
        else:
            for j in range(int(csv_read.loc[i][6])-1):
                labeli.append(0)#事件前面的O
            labeli.append(1)#事件开始
            for j in range(int(csv_read.loc[i][6]),int(csv_read.loc[i][7])):
                labeli.append(2)#事件继续
            if (str(csv_read.loc[i][8])=='nan'):#句子中只有一个事件
                for j in range(int(csv_read.loc[i][7]),int(csv_read.loc[i][5])):
                    labeli.append(0)#事件后面的O
            else:#不止一个事件

                    for j in range(int(csv_read.loc[i][7]),int(csv_read.loc[i][8])-1):
                        labeli.append(0)#两个事件中间的O
                    labeli.append(1)#第二个事件开始
                    for j in range(int(csv_read.loc[i][8]),int(csv_read.loc[i][9])):
                        labeli.append(2)#第二个事件继续
                    if (str(csv_read.loc[i][10])=='nan'):#句子中只有两个事件
                        for j in range(int(csv_read.loc[i][9]),int(csv_read.loc[i][5])):
                            labeli.append(0)#第二个事件结束后的0
                    else:#有三个事件
                        for j in range(int(csv_read.loc[i][9]),int(csv_read.loc[i][10])-1):
                            labeli.append(0)#第二个和第三个事件中间的0
                        labeli.append(1)#第三个事件开始
                        for j in range(int(csv_read.loc[i][10]),int(csv_read.loc[i][11])):
                            labeli.append(2)#第三个事件继续
                        if (str(csv_read.loc[i][12])=='nan'):#最多只有三个事件
                            for j in range(int(csv_read.loc[i][11]),int(csv_read.loc[i][5])):
                                labeli.append(0)#第三个事件结束后的0
                        else:#有四个事件
                            for j in range(int(csv_read.loc[i][11]),int(csv_read.loc[i][12])-1):
                                labeli.append(0)#第三个和第四个事件中间的0
                            labeli.append(1)#第四个事件开始
                            for j in range(int(csv_read.loc[i][12]),int(csv_read.loc[i][13])):
                                labeli.append(2)#第四个事件继续
                            #最多只有三个事件
                            for j in range(int(csv_read.loc[i][13]),int(csv_read.loc[i][5])):
                                labeli.append(0)#第三个事件结束后的0
        for j in range(len(labeli),max_length):
            labeli.append(0)#补足长度
        label.append(labeli)
    #print(label)
    label_array = np.asarray(label)
    np.save('y_train',label_array)
    #print(label_array.shape)
    return label_array


#BIRNN
class bilstm_crf(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(bilstm_crf, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size,hidden_size,num_layers,batch_first = True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
        self.crf = CRF(num_classes,batch_first=True)

    def forward(self, x):
        #set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        #out, _ = self.lstm(x,(h0,c0))#out:tensor of shape (batch_size,seq_length,hidden_size*2
        out, _ = self.lstm(x,h0)#out:tensor of shape (batch_size,seq_length,hidden_size*2

        #decode the hidden state of the last time step
        out = self.fc(out)

        return out




def validation(validation_loader):
    #valid the model
    with torch.no_grad():
        correct = 0
        total = 0
        for sentences, labels in validation_loader:
            sentences = sentences.reshape(-1,sequence_length, input_size).to(device)

            outputs = model(sentences)
            outputs = model.crf.decode(outputs)
            labels = labels.numpy().tolist()
            total += sentences.size(0)
            for i in range(len(labels)):
                if (outputs[i]==labels[i]):
                    correct+=1
            #print(total,correct)

        print('Test Accuracy of the model : {} %'.format(100 * correct / total))

def train(train_loader,validation_loader):
    #train the model
    total_step = len(train_loader)
    print('total_step:',total_step)
    for epoch in range(num_epochs):
        for i, (sentence,labels) in enumerate(train_loader):
            sentence = sentence.reshape(-1,sequence_length, input_size).to(device)
            labels = labels.to(device)
            #print(train_loader,sentence.size(),labels.size())
            #forward pass
            y_hat = model(sentence)
            #print(y_hat.size())
            loss = -model.crf(y_hat, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs, i+1,total_step, loss.item()))
        validation(validation_loader)
    #save the model checkpoint

    torch.save(model,'./model.pth')

def test(test_file,test_loader):
    validation(test_loader)
    output = []

    with torch.no_grad():

        for sentence,_ in test_loader:
            sentence = sentence.reshape(-1,sequence_length, input_size).to(device)
            y_hat = model(sentence)
            y_hat = model.crf.decode(y_hat)#(batch_size,sequence_length)
            for i in range (len(y_hat)):
                output.append(y_hat[i])
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

    print(result)
    test_csv = pd.read_csv(test_file,encoding='utf8')
    f=open('result.csv','a+')
    csv_write = csv.writer(f)
    csv_write.writerow(['article','sentence','keyword','cause/effect','content'])
    for i in range(len(result)):
        if(len(result[i])==0):#该句中没有事件
            row=[]
            for j in range(5):
                row.append(test_csv.loc[i][j])
            csv_write.writerow(row)
        else:
            row=[]
            for j in range(5):
                row.append(test_csv.loc[i][j])
            for j in range(0,len(result[i]),2):
                row.append(result[i][j]+1)#+1对应人手工标注
                row.append(result[i][j+1]+1)
            for j in range(0,len(result[i]),2):
                row.append(test_csv.loc[i][4][result[i][j]:result[i][j+1]+1])#文字内容
            csv_write.writerow(row)
    f.close()

def main():
    #model = bilstm_crf(input_size,hidden_size, num_layers,num_classes).to(device)
    model = torch.load('./model.pth')
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    train_loader,validation_loader,test_loader=construct_dataset()
    train(train_loader,validation_loader)
    test('new_extraction_event_financialnews.com.cn.csv',test_loader)
