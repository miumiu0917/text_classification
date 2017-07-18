import csv
import numpy as np

UNKNOWN_ID = 0
UNKNOWN = 'UNK_'
NUM_CATEGORY = 6

class InputData(object):

  def __init__(self, batch_size=64):
    self.batch_size = batch_size
    self.chars_dict = self.__create_char_dict()
    self.max_len = self.__max_len()
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


  def __read_data(self, data_path):
    data = []
    with open(data_path, 'r') as f:
      reader = csv.reader(f)
      for line in reader:
        data.append([line[0], len(line[1]), line[1]])
    return data


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
    lens = [len(e[1]) for e in batch]
    texts = [e[1] for e in batch]
    self.idx += self.batch_size
    return labels, lens, texts
  

  def __one_hot_vector(self, index):
    one_hot = np.zeros(NUM_CATEGORY)
    one_hot[int(index)] = 1
    return one_hot