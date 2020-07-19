#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
sys.path.append('./src')
from flask import Flask, render_template, request
from src.model import PoetryModel
from src.generate import generate, generate_acrostic
from src.config import Config
import numpy as np
import torch


class PoetryGenerator:
    def __init__(self):
        self.data = np.load(Config['numpy_data_path'][1:], allow_pickle=True)
        self.word2index = self.data['word2index'].item()
        self.index2word = self.data['index2word'].item()
        self.model = PoetryModel(len(self.word2index), Config['embedding_dim'], Config['hidden_dim'])
        self.model.load_state_dict(torch.load(Config['model_path'][1:]))
        if Config['use_gpu']:
            self.model.to(Config['device'])

    # 续写诗
    def gen(self, start_words, prefix_words=None):
        return generate(self.model, start_words, self.word2index, self.index2word, prefix_words)

    # 藏头诗
    def gen_acrostic(self, start_words, prefix_words=None):
        return generate_acrostic(self.model, start_words, self.word2index, self.index2word, prefix_words)


Generator = PoetryGenerator()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return 'hello world'


@app.route('/generator', methods=['POST', 'GET'])
def generator():
    if request.method == 'POST' and request.form['start_words']:
        if request.form['type'] == '1':
            poetry = Generator.gen(request.form['start_words'], request.form['prefix_words'])
        elif request.form['type'] == '2':
            poetry = Generator.gen_acrostic(request.form['start_words'], request.form['prefix_words'])

        return render_template('generate.html', poetry=poetry)
    else:

        return render_template('generate.html', poetry="")


if __name__ == '__main__':
    app.run(debug=True)

