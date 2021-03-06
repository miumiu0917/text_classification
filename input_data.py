import csv
import random
import numpy as np

UNKNOWN_ID = 0
UNKNOWN = 'UNK_'

class InputData(object):

  def __init__(self, batch_size=64, num_category=6, train=True):
    self.num_category = num_category
    self.batch_size = batch_size
    if train:
      self.chars_dict = self.__create_char_dict()
    else:
      self.chars_dict = self.__load_char_dict()
    self.num_chars = len(list(self.chars_dict.keys()))
    self.max_len = self.__max_len()
    # 以下はtrainモードのみ
    if train:
      self.train_data = self.__read_data('data/train.csv')
      test_data = self.__read_data('data/test.csv')
      self.test_label, self.test_lens, self.test_text = self.__shaping_test_data(test_data)
      self.idx = 0
    

  def __create_char_dict(self):
    chars = set()
    with open('data/raw.csv', 'r') as f:
      reader = csv.reader(f)
      for line in reader:
        chars |= set(list(line[1]))
    # 順序の一貫性を保つために配列に変換
    chars = list(chars)
    chars = [UNKNOWN] + chars
    # あとで使えるようにファイルに保存
    with open('data/chars.txt', 'w') as f:
      for c in chars:
        f.write(c + "\n")
    return {c:i for i, c in enumerate(chars)}


  def __load_char_dict(self):
    with open('data/chars.txt', 'r') as f:
      return {c.rstrip("\n"):i for i, c in enumerate(f)}

  def __read_data(self, data_path):
    data = []
    with open(data_path, 'r') as f:
      reader = csv.reader(f)
      for line in reader:
        ids = self.sentence_to_vector(line[1])
        data.append([line[0], len(line[1]), ids])
    return data


  def sentence_to_vector(self, sentence):
    sentence = list(sentence)
    sentence = [c if c in self.chars_dict else UNKNOWN for c in sentence]
    ids = [self.chars_dict[c] for c in sentence]
    ids = ids + [0 for _ in range(self.max_len - len(ids))]
    return ids
    

  def __max_len(self):
    with open('data/raw.csv', 'r') as f:
      reader = csv.reader(f)
      return max([len(line[1]) for line in reader])

  
  def __shaping_test_data(self, test_data):
    test_label = [self.__one_hot_vector(e[0]) for e in test_data]
    test_lens = [e[1] for e in test_data]
    test_text = [e[2] for e in test_data]
    return test_label, test_lens, test_text

  
  def next_batch(self):
    if self.idx + self.batch_size > len(self.train_data):
      self.idx = 0
      random.shuffle(self.train_data)
    batch = self.train_data[self.idx:self.idx+self.batch_size]
    labels = [self.__one_hot_vector(e[0]) for e in batch]
    lens = [e[1] for e in batch]
    texts = [e[2] for e in batch]
    self.idx += self.batch_size
    return labels, lens, texts
  

  def __one_hot_vector(self, index):
    one_hot = np.zeros(self.num_category)
    one_hot[int(index)] = 1
    return one_hot