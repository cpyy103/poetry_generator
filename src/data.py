#!/usr/bin/python
# -*- coding:utf-8 -*-
# 处理原数据即获取处理完的数据

import json
import os
import numpy as np
import re
from src.config import Config

from src.langconv import *


def parse_raw_data(author=Config['author'], constrain=Config['constrain'], src=Config['data_path'], category=Config['category']):
    '''
    处理源json数据，返回诗歌内容
    :param author: 作者名字，是否指定诗歌的作者
    :param src: 数据保存路径
    :param category: 数据类型，主要有唐诗 poet.tang 和宋词 poet.song
    :return: list
        ['公子申敬爱，携朋翫物华。人是平阳客，地即石崇家。水文生旧浦，风色满新花。日暮连归骑，长川照晚霞。',
        '高门引冠盖，下客抱支离。绮席珍羞满，文场翰藻摛。蓂华彫上月，柳色蔼春池。日斜归戚里，连骑勒金羁。',
        '今夜可怜春，河桥多丽人。宝马金为络，香车玉作轮。连手窥潘掾，分头看洛神。重城自不掩，出向小平津。',
        ...]
    '''


    def traditional2simplified(sentence):
        # 繁体转简体
        return Converter('zh-hans').convert(sentence)

    def simplified2traditional(sentence):
        # 简体转繁体
        return Converter('zh-hant').convert(sentence)

    def sentence_parse(para):
        # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。
        # （「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
        # 处理类似上面的有其他不需要的字符的诗句
        # 返回纯诗词句

        result, number = re.subn('（.*）', '', para)
        result, number = re.subn('{.*}', '', result)
        result, number = re.subn('《.*》', '', result)
        result, number = re.subn('《.*》', '', result)
        result, number = re.subn('[\]\[]', '', result)
        r = ''
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u'。。', u'。', r)
        return r

    def handle_json(file):
        result = []
        with open(file, encoding='utf-8') as f:
            data = json.loads(f.read())

        for poetry in data:
            pdata = ''
            # 是否为特定作者
            # 如果不是特定作者或没有指定作者，则继续后面的程序
            if author != None and poetry.get('author') != author:
                continue

            p = poetry.get('paragraphs')
            # 是否限定诗词的每句长度
            flag = False
            for s in p:
                sp = re.split('[，！。]', s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue

            for sentence in poetry.get('paragraphs'):
                pdata += sentence
            pdata = sentence_parse(pdata)
            if pdata != '':
                result.append(traditional2simplified(pdata))
        return result

    data = []
    for file_name in os.listdir(src):
        if file_name.startswith(category):
            data.extend(handle_json(src+file_name))
    return data


def pad_sequence(sequences, max_len=None, dtype='int32', padding='post', truncating='post', value=0):
    '''
    填充语句
    将长度小于max_len的句子用空格补充，使其长度为max_len
    将长度大于max_len的句子进行截断，使其长度为max_len
    :param sequences: list 不同句子
    :param max_len: 每句的最大长度
    :param dtype: 返回句子的数据类型
    :param padding: 'pre'或 'post', pre：在句子前面补充空格， post：在句子后面补充
    :param truncating: 'pre'或'post', pre：截断并舍弃长句子的前面超出长度的部分，post:截断后面部分
    :param value: 用于补充的值
    :return: X numpy 矩阵 size = ( number_of_sequence x max_len )
    '''
    print('padding data...')
    length = []
    for s in sequences:
        length.append(len(s))

    if max_len is None:
        max_len = max(length)

    x = (np.ones((len(sequences), max_len))*value).astype(dtype)
    for index, s in enumerate(sequences):
        if not length[index] or length[index] == max_len:
            continue
        if length[index] > max_len:
            if truncating == 'pre':
                x[index, -max_len:] = s[-max_len:]
            elif truncating == 'post':
                x[index, :max_len] = s[:max_len]
            else:
                raise ValueError('truncating error -  %s' % truncating)

        elif length[index] < max_len:
            if padding == 'pre':
                x[index, -length[index]:] = s
            elif padding == 'post':
                x[index, :length[index]] = s
            else:
                raise ValueError('padding error - %s' % padding)
    return x


def get_data():
    '''
    返回训练的数据
    :return: data: numpy数组，每一行是一首诗，每个数字是对应汉字的对应下标
    :return: word2index: dict, 每个汉字对应数字
    :return: index2word: dict, 每个数字对应汉字
    '''
    print('getting data...')
    if os.path.exists(Config['numpy_data_path']):
        data = np.load(Config['numpy_data_path'], allow_pickle=True)
        data, word2index, index2word = data['data'], data['word2index'].item(), data['index2word'].item()
        return data, word2index, index2word

    # 如果不存在处理好的数据
    data = parse_raw_data()
    words = set([word for sentence in data for word in sentence])
    word2index = {word: index for index, word in enumerate(words)}
    word2index['<START>'] = len(word2index)   # 起始符
    word2index['<EOP>'] = len(word2index)     # 终止符
    word2index['<SPACE>'] = len(word2index)   # 空格
    index2word = {index: word for word, index in list(word2index.items())}

    # 为每首诗歌加上起始终止符
    for i in range(len(data)):
        data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    # 将汉字转变为数字
    data = [[word2index[word] for word in sentence] for sentence in data]

    # 进行padding
    padding_data = pad_sequence(data, max_len=Config['max_len'],
                                padding='pre', truncating='post', value=word2index['<SPACE>'])

    # 保存成二进制文件
    np.savez_compressed(Config['numpy_data_path'], data=padding_data, word2index=word2index, index2word=index2word)

    return padding_data, word2index, index2word


if __name__ == '__main__':
    # data = parse_raw_data(category='poet.tang.4000')
    # print(data)
    padding_data, word2index, index2word = get_data()

    test_data = padding_data[3]
    poetry = ''.join([index2word[index] for index in test_data])
    print(poetry)
    print(padding_data.shape)
    print(word2index)
    print(len(word2index))




















