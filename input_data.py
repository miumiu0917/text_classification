import csv

class InputData(object):

  def __init__(self):
    self.chars_dict = self.__create_char_dict()
    self.max_len = self.__max_len()
    print(self.max_len)
    

  def __create_char_dict(self):
    chars = set()
    with open('data/raw.csv', 'r') as f:
      reader = csv.reader(f)
      for line in reader:
        chars |= set(list(line[1]))
    # 順序の一貫性を保つために配列に変換
    chars = list(chars)
    # あとで使えるようにファイルに保存
    with open('data/chars.txt', 'w') as f:
      for c in chars:
        f.write(c + "\n")
    return {c:i for i, c in enumerate(chars)}

  def __max_len(self):
    with open('data/raw.csv', 'r') as f:
      reader = csv.reader(f)
      return max([len(line[1]) for line in reader])