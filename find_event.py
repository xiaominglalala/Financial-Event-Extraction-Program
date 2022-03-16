# coding: utf-8
# File: find_event.py
# Date: 2020-1-12

import re, jieba, os, csv
import jieba.posseg as pseg
from pyltp import SentenceSplitter
class CausalityExractor():
    def __init__(self):
        pass

    '''1由果溯因配套式'''
    def ruler1(self, sentence):
        datas = list()
        word_pairs =[['之所以', '是因为'], ['之?所以', '由于'], ['之?所以', '缘于']]
        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)

            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][3]
                data['effect'] = result[0][1]
                datas.append(data)
        if datas:
            return 1,datas[0]
        else:
            return 0,{}
    '''2由因到果配套式'''
    def ruler2(self, sentence):
        '''
        conm1:〈因为,从而〉、〈因为,为此〉、〈既[然],所以〉、〈因为,为此〉、〈由于,为此〉、〈只有|除非,才〉、〈由于,以至[于]>、〈既[然],却>、
        〈如果,那么|则〉、<由于,从而〉、<既[然],就〉、〈既[然],因此〉、〈如果,就〉、〈只要,就〉〈因为,所以〉、 <由于,于是〉、〈因为,因此〉、
         <由于,故〉、 〈因为,以致[于]〉、〈因为,因而〉、〈由于,因此〉、<因为,于是〉、〈由于,致使〉、〈因为,致使〉、〈由于,以致[于] >
         〈因为,故〉、〈因[为],以至[于]>,〈由于,所以〉、〈因为,故而〉、〈由于,因而〉
        conm1_model:<Conj>{Cause}, <Conj>{Effect}
        '''
        datas = list()
        word_pairs =[['因为', '从而'], ['既然?', '所以'],
                    ['由于', '以至于?'],
                     ['由于', '从而'],
                    ['既然?', '因此'],
                    ['因为', '所以'], ['由于', '于是'],
                    ['因为', '因此'], ['由于', '故'],
                    ['因为', '因而'], ['由于', '因此'],
                    ['因为', '于是'],
                    ['由于', '所以'], ['由于', '因而'],
        #extra
                    ['因为?','故']
                    ]

        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c|n]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][1]
                data['effect'] = result[0][3]
                datas.append(data)
        if datas:
            return 1,datas[0]
        else:
            return 0,{}
    '''3由因到果居中式明确'''
    def ruler3(self, sentence): #-->r'(.*)[,，]+.*()/[p|c]+\s(.*)'
        pattern = re.compile(r'(.*)[,，]+.*(于是|所以|致使|以致于?|因此|以至于?|从而|因而)/[p|c|v]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
            return 1, data
        return 0, data
    '''4由因到果居中式精确'''
    def ruler4(self, sentence):
        pattern = re.compile(r'(.*)\s+(已致|导致|指引|使|促成|造成|造就|促使|酿成|引发|促进|引起|引来|促发|引致|诱发|推动|招致|致使|滋生|作用|使得|决定|令人|带来|触发|归因于)/[d|v]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
            return 1, data
        return 0, data
    '''5由因到果前端式模糊'''
    def ruler5(self, sentence):
        pattern = re.compile(r'\s?(因为|因|凭借|由于)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]
            return 1, data
        return 0, data

    '''6由因到果居中式模糊'''
    def ruler6(self, sentence):
        pattern = re.compile(r'\s(.*)(以免|以便)/[c|d]+\s(.*)')
        result = pattern.findall(sentence)

        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
            return 1, data
        return 0, data

    '''7由因到果前端式精确'''
    def ruler7(self, sentence):
        pattern = re.compile(r'\s?(只要)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]
            return 1, data
        return 0, data

    '''8由果溯因居中式模糊'''
    def ruler8(self, sentence):
        pattern = re.compile(r'(.*)(取决于|缘于|在于|出自|来自|发自|源于)[p|c|v]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]
            return 1, data
        return 0, data

    '''9 名词性匹配'''
    def ruler9(self,sentence):
        '''的原因是，'''
        pattern = re.compile(r'(.*)的/uj 原因/n 是/v+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = "的原因是"
            data['cause'] = result[0][1]
            data['effect'] = result[0][0]
            return 1,data
        return 0,data

    '''10 名词性匹配'''
    def ruler10(self,sentence):
        '''的结果是'''
        pattern = re.compile(r'(.*)的/uj 结果n 是/v+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = "的结果是"
            data['cause'] = result[0][0]
            data['effect'] = result[0][1]
            return 1, data
        return 0, data


    '''抽取主函数'''
    def extract_triples(self, sentence):

        result = dict()
        '''1，9，10比较明确'''
        match, result = self.ruler1(sentence)
        if match==1:
            return result

        match, result = self.ruler9(sentence)
        if match==1:
            return result

        match, result = self.ruler10(sentence)
        if match==1:
            return result

        match, result = self.ruler2(sentence)
        if match==1:
            return result

        match, result = self.ruler3(sentence)
        if match==1:
            return result

        match, result = self.ruler4(sentence)
        if match==1:
            return result

        match, result = self.ruler5(sentence)
        if match==1:
            return result

        match, result = self.ruler6(sentence)
        if match==1:
            return result

        match, result = self.ruler7(sentence)
        if match==1:
            return result

        match, result = self.ruler8(sentence)
        if match==1:
            return result

        return result

    '''抽取主控函数'''
    def extract_main(self, sent):
        #分词+词性标注
        sent = ' '.join([word.word + '/' + word.flag for word in pseg.cut(sent)])
        result = self.extract_triples(sent)
        return result

'''测试'''
def test():
    extractor = CausalityExractor()
    target = open("raw_event_sent.txt",'w',encoding='utf-8')

    with open('qihuowang_content_with_keywords.csv','r',encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            #分句
            datas = [sentence for sentence in SentenceSplitter.split(row[4]) if sentence]

            for j, data in enumerate(datas):
                result = extractor.extract_main(data)
                if result:
                    cause = ''.join([word.split('/')[0] for word in result['cause'].split(' ') if word.split('/')[0]])
                    effect = ''.join([word.split('/')[0] for word in result['effect'].split(' ') if word.split('/')[0]])
                    triger = result['tag'].split('-')


                    # save i: document id, j:sentence id, and indexes
                    ids = [str(i),str(j)]

                    # for finditer to work, make sure cause,effect,triger don't have '('or')'
                    #if '(' in cause or '(' in effect or ')'in cause or ')' in effect:
                        #continue
                    try:
                        flag = 0
                        for it in re.finditer(data,row[4]):
                            ids = ids + [str(it.span()[0]), str(it.span()[1])]
                            flag = 1
                            break
                        if flag==0:
                            ids = ids + ['-1', '-1']

                        #get the index
                        flag = 0
                        for it in re.finditer(cause,data):
                            ids = ids + [str(it.span()[0]), str(it.span()[1])]
                            flag = 1
                            break
                        if flag==0:
                            ids = ids + ['-1','-1']

                        flag = 0
                        for it in re.finditer(effect, data):
                            ids = ids + [str(it.span()[0]), str(it.span()[1])]
                            flag = 1
                            break
                        if flag == 0:
                            ids = ids + ['-1', '-1']

                        for w in triger:
                            flag = 0
                            for it in re.finditer(w, data):
                                ids = ids + [str(it.span()[0]), str(it.span()[1])]
                                flag = 1
                                break
                            if flag == 0:
                                ids = ids + ['-1', '-1']

                        target.write(" ".join(ids))
                        target.write('\n')
                    except:
                        continue

    target.close()

test()