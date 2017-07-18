import mojimoji
import random
import csv


CATEGORY_DICT = {
  0: [0, 500]
  , 1: [500, 1000]
  , 2: [1000, 4999]
  , 3: [5000, 10000]
  , 4: [10000, 50000]
  , 5: [50000, 100000]
  }


def main():
  first = read_from_file('data/candidate/first.txt')
  last = read_from_file('data/candidate/last.txt')
  money_first = read_from_file('data/candidate/money_first.txt')
  money_last = read_from_file('data/candidate/money_last.txt')
  with open('data/raw.csv', 'w') as f:
    with open('data/train.csv', 'w') as train:
      with open('data/test.csv', 'w') as test:
        writer_f = csv.writer(f, lineterminator='\n')
        writer_train = csv.writer(train, lineterminator='\n')
        writer_test = csv.writer(test, lineterminator='\n')
        for i in range(50000):
          category, money_ = money()
          text = random.choice(first) + random.choice(money_first) + money_ + random.choice(money_last) + random.choice(last)
          writer_f.writerow([category, text])
          writer_train.writerow([category, text]) if i < 40000 else writer_test.writerow([category, text])
          

def money():
  category = random.choice(list(CATEGORY_DICT.keys()))
  range_ = CATEGORY_DICT[category]
  return category, convert(str(random.randrange(range_[0], range_[1])))

def read_from_file(file_path):
  with open(file_path, 'r') as f:
    return [line.rstrip("\n") for line in f]

def convert(money):
  fifty_fifty = lambda: random.randint(0,1) % 2 == 0
  comma = lambda s: "{:,}".format(int(money))
  han2zen = lambda s: mojimoji.han_to_zen(s)
  unit = lambda s: s + '円' if fifty_fifty() else '￥' + s
  if fifty_fifty():
    money = comma(money)
  if fifty_fifty():
    money = han2zen(money)
  money = unit(money)
  return money


if __name__ == '__main__':
  main()