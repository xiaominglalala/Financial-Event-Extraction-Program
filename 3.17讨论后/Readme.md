# 关于数据格式

1. 第一列为文章编号，由李桐改过的代码得到

2. 第二列为句子编号 ，由李桐改过的代码得到

3. 第三列为关键词

4. 第四列为cause/effect，一对事件紧邻，cause在上，effect在下

5. 第五列为cause/effect的内容，由李桐改过的代码直接提取得到

6. 第六列为第五列的内容的长度

7. 第七列为事件1起始位置（人工标注出的）

8. 第八列为事件1终止位置

9. 第九列为事件2起始位置

10. 第十列为事件2终止位置

11. 第十一列为事件3起始位置

12. 第十二列为事件3终止位置（在我标的数据里面一句话最多只有三个事件，如果你们的有更多，那我可能要再改一下代码）

    

见数据格式.csv

训练集、验证集、测试集均为此格式，测试集可以没有label

# 关于pytorch.py

## 使用方法

1. 用create_embedding_x函数构建word embedding，分别构建训练集，验证集和测试集的.npy文件保存起来
2. 用create_label_y函数构建labels，分别构建训练集，验证集和测试集的.npy文件保存起来
3. main（）函数，用train（）训练后test（）预测



## 各函数功能

### create_embedding_x(input_file,range1,range2)

由于如果每次训练和测试都用bert提取一遍embedding的话会非常耗时而且耗内存，所以我是先一次性提取出来，将embedding保存在numpy文件中，之后只要导入就行

可先分别生成training set, validation set, test set

#### 输入

input_file：即如数据格式.csv

range1/range2：得到的embedding是从csv中第range1行到第range2行的content列的embedding

#### 输出

返回维度为（range2-range1,100,768）的矩阵并将其保存为.npy文件

（bert得到的embedding为768维，句子长度限制在100，不足会补零）

### create_label_y(input_file)

#### 输入

input_file格式同数据格式.csv

range1/range2：得到的label是从csv中第range1行到第range2行的

#### 输出

维度为（range2-range1,100）的矩阵并保存



这个函数写的比较挫，之后可能要重写一下



### class bilstm_crf(nn.Module):

定义bilstm_crf类

### train（）

训练模型

从train_loader取数据

每10 steps输出一次loss

每个epoch做一次validation

最后保存模型

### validation（）

验证模型

从validation_loader取数据

对比validation set中的label和模型预测的

输出正确率

### test (test_file)

测试/预测

#### 输入

test_file格式同数据格式.csv，为测试集，可无label（即同数据格式.csv的前5列）

####　输出

输出正确率

生成result.csv

前五列为test_file的前5列，后面为事件索引，最后是事件文字

见result.csv （utf8格式，可用excel->数据->获取外部数据->自文本 打开）

## 关于数据导入construct_dataset()

1. 用np.load导入训练集、验证集、测试集，用mmap_mode='r'可以节省内存
2. 转化为torch tensor
3. 构建train_loader, validation_loader, test_loader并返回



# model.pth

目前训练得到的模型

可通过model = torch.load('./model.pth')直接用

# triples.csv

目前模型对金融新闻网前25000条数据提取到的因果对

