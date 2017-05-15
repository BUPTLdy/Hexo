---
layout:     post
title:      "Shopping Reviews sentiment analysis"
subtitle:   "购物评论情感分析的实现"
date:       2016-07-20 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - LSTM
    - Deep Learning
    - NLP
---

![](http://7xritj.com1.z0.glb.clouddn.com/16-7-18/19781698.jpg)
<!--more-->
情感分析是一种常见的自然语言处理（NLP）方法的应用，特别是在以提取文本的情感内容为目标的分类方法中。通过这种方式，情感分析可以被视为利用一些情感得分指标来量化定性数据的方法。尽管情绪在很大程度上是主观的，但是情感量化分析已经有很多有用的实践，比如企业分析消费者对产品的反馈信息，或者检测在线评论中的差评信息。

最简单的情感分析方法是利用词语的正负属性来判定。句子中的每个单词都有一个得分，乐观的单词得分为+1，悲观的单词则为-1。然后我们对句子中所有单词得分进行加总求和得到一个最终的情感总分。很明显，这种方法有许多局限之处，最重要的一点在于它忽略了上下文的信息。例如，在这个简易模型中，因为“not”的得分为-1，而“good”的得分为 +1，所以词组“not good”将被归类到中性词组中。但是“not good”通常是消极的。

另外一个常见的方法是将文本视为一个“词袋”。我们将每个文本看出一个`1xN`的向量，其中N表示文本词汇的数量。该向量中每一列都是一个单词，其对应的值为该单词出现的频数。例如，词组“bag of bag of words”可以被编码为`[2, 2, 1]`。这些数据可以被应用到机器学习分类算法中（比如罗吉斯回归或者支持向量机），从而预测未知数据的情感状况。需要注意的是，这种有监督学习的方法要求利用已知情感状况的数据作为训练集。虽然这个方法改进了之前的模型，但是它仍然忽略了上下文的信息和数据集的规模情况。

## Word2Vec and Doc2Vec

谷歌开发了一个叫做Word2Vec的方法，该方法可以在捕捉语境信息的同时压缩数据规模。Word2Vec实际上是两种不同的方法：Continuous Bag of Words (CBOW) 和 Skip-gram。CBOW的目标是根据上下文来预测当前词语。Skip-gram刚好相反：根据当前词语来预测上下文。这两种方法都利用人工神经网络作为它们的分类算法。起初，每个单词都是一个随机的 N 维向量。经过训练之后，该算法利用 CBOW 或者 Skip-gram 的方法获得了每个单词的最优向量。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-24/80870876.jpg)
在上图中 $w(t)$ 表示当前的词汇，$w(t-2)$ ， $w(t-1)$ 等表示上下文词汇。

现在这些词向量已经捕捉到上下文的信息。我们可以利用基本代数公式来发现单词之间的关系（比如，“国王”-“男人”+“女人”=“王后”）。这些词向量可以代替词袋用来预测未知数据的情感状况。该模型的优点在于不仅考虑了语境信息还压缩了数据规模（通常情况下，词汇量规模大约在300个单词左右而不是之前模型的100000个单词）。因为神经网络可以替我们提取出这些特征的信息，所以我们仅需要做很少的手动工作。

## 使用SVM和Word2Vec进行情感分类

我们使用的训练数据是网友[苏剑林](http://spaces.ac.cn/archives/3414/)收集分享的两万多条中文标注语料，涉及六个领域的评论数据。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-25/48028586.jpg)
我们随机正负这两组数据中抽取样本，构建比例为8：2的训练集和测试集。随后，我们对训练集数据构建Word2Vec模型，其中分类器的输入值为推文中所有词向量的加权平均值。word2vec工具和svm分类器分别使用python中的[gensim](https://radimrehurek.com/gensim/)库和[sklearn](http://scikit-learn.org/)库。

加载文件，并分词

```python
# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    pos=pd.read_excel('data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    #print pos['words']
    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train,x_test

```

计算词向量，并对每个评论的所有词向量取均值作为每个评论的输入
```python
#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)

    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)

    np.save('svm_data/train_vecs.npy',train_vecs)
    print train_vecs.shape
    #Train word2vec on test tweets
    imdb_w2v.train(x_test)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print test_vecs.shape

```

训练svm模型

```python
##训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print clf.score(test_vecs,y_test)
```
在没有创建任何类型的特性和最小文本预处理的情况下，我们利用Scikit-Learn构建的简单线性模型的预测精度为80%左右。有趣的是，删除标点符号会影响预测精度，这说明Word2Vec模型可以提取出文档中符号所包含的信息。处理单独的单词，训练更长时间，做更多的数据预处理工作，和调整模型的参数都可以提高预测精度。用svm分类有一个缺点是，我们把每个句子的词向量求平均丢失了句子词语之间的顺序信息。


## 使用LSTM和Word2Vec进行情感分类

人类的思维不是每时每刻都是崭新的，就像你阅读一篇文章时，你理解当前词语的基础是基于对之前词语的理解，人类的思维是能保持一段时间的。传统的人工神经网络，并不能模拟人类思维具有记忆性这一特征，例如，你想要分类电影在某一时间点发生了什么事情，使用传统的人工神经网络并不能清楚的表现出之前出现的镜头对当前镜头的提示。循环神经网络能够很好的处理这个问题。
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-7-8/34980285.jpg)
</center>
RNN相对于传统的神经网络，它允许我们对向量序列进行操作：输入序列、输出序列、或大部分的输入输出序列。如下图所示，每一个矩形是一个向量，箭头则表示函数（比如矩阵相乘）。输入向量用红色标出，输出向量用蓝色标出，绿色的矩形是RNN的状态（下面会详细介绍）。从做到右：（1）没有使用RNN的Vanilla模型，从固定大小的输入得到固定大小输出（比如图像分类）。（2）序列输出（比如图片字幕，输入一张图片输出一段文字序列）。（3）序列输入（比如情感分析，输入一段文字然后将它分类成积极或者消极情感）。（4）序列输入和序列输出（比如机器翻译：一个RNN读取一条英文语句然后将它以法语形式输出）。（5）同步序列输入输出（比如视频分类，对视频中每一帧打标签）。我们注意到在每一个案例中，都没有对序列长度进行预先特定约束，因为递归变换（绿色部分）是固定的，而且我们可以多次使用。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-25/61310852.jpg)

单纯循环神经网络因为无法处理随着递归，权重指数级爆炸或消失的问题（Vanishing gradient problem），难以捕捉长期时间关联；而结合不同的LSTM可以很好解决这个问题。

LSTM 全称叫 Long Short Term Memory networks，它和传统 RNN 唯一的不同就在与其中的神经元（感知机）的构造不同。传统的 RNN 每个神经元和一般神经网络的感知机没啥区别，但在 LSTM 中，每个神经元是一个“记忆细胞”，细胞里面有一个“输入门”（input gate）, 一个“遗忘门”（forget gate）， 一个“输出门”（output gate）。

这个设计的用意在于，能够使得LSTM维持两条线，一条明线：当前时刻的数据流（包括其他细胞的输入和来自数据的输入）；一条暗线：这个细胞本身的记忆流。两条线互相呼应，互相纠缠，就像佛祖青灯里的两根灯芯。典型的工作流如下：在“输入门”中，根据当前的数据流来控制接受细胞记忆的影响；接着，在“遗忘门”里，更新这个细胞的记忆和数据流；然后在“输出门”里产生输出更新后的记忆和数据流。LSTM 模型的关键之一就在于这个“遗忘门”， 它能够控制训练时候梯度在这里的收敛性（从而避免了 RNN 中的梯度 vanishing/exploding问题），同时也能够保持长期的记忆性。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-25/14675212.jpg)

### 实现步骤:

加载训练文件并分词
```python
#加载训练文件
def loadfile():
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    pos=pd.read_excel('data/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined,y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

```
创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引

```python
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    model.save('lstm_data/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

```

训练网络，并保存模型，其中LSTM的实现采用Python中的[keras](http://keras.io/)库

```python
def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print 'Defining a Simple Keras Model...'
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print "Train..."
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)

    print "Evaluate..."
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print 'Test score:', score


#训练模型，并保存
def train():
    print 'Loading Data...'
    combined,y=loadfile()
    print len(combined),len(y)
    print 'Tokenising...'
    combined = tokenizer(combined)
    print 'Training a Word2vec model...'
    index_dict, word_vectors,combined=word2vec_train(combined)
    print 'Setting up Arrays for Keras Embedding Layer...'
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print x_train.shape,y_train.shape
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


```

### 结果分析
 使用LSTM网络在测试集上的准确率为92%，比用SVM分类提高了不少。


## 代码地址

[https://github.com/BUPTLdy/Sentiment-Analysis](https://github.com/BUPTLdy/Sentiment-Analysis)

## 参考

[http://www.15yan.com/story/huxAyyeuYAj/](http://www.15yan.com/story/huxAyyeuYAj/)

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
