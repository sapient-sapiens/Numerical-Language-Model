import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
# WE will use the reverse polish notation

digits = [str(i) for i in range(10)]
operations = ['+', '*'] # Delete dividing for now, let's not deal with decimals

result = ['=']
space = [' ']
eot = ['.']
# Dedicated PAD token distinct from EOT to avoid leaking '.' during training
pad = ['_']

vocab = digits + operations + result + space + eot + pad

stoi = {c:i for i, c in enumerate(vocab)}
itos = {i:c for i, c in enumerate(vocab)}



d_embed = 256
embedding = nn.Embedding(len(vocab), d_embed)

class Calculations(Dataset):
    def __init__(self, dataset_size, p):
        self.p = p
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def generate_word(self):
        # return a NUMBER STRING (not list), since evaluate() builds a string then resets it
        word = random.choice(digits)
        while random.random() > self.p:
            word += random.choice(digits)
        return word

    def evaluate(self, s):
        # s is an iterable of chars (your seq list will contain digits, ops, and spaces)
        word = ''         # current number being built
        stack = []
        for c in s:
            if c in digits:
                word += c
            elif c == ' ':
                if word:                # only push if we actually built a number
                    stack.append(int(word))
                    word = ''           # reset (strings don't have .clear())
            else:
                # c is an operator
                if word:
                    stack.append(int(word))
                    word = ''
                assert len(stack) >= 2
                b = stack.pop()
                a = stack.pop()
                if c == '+':            # use == for strings, not `is`
                    stack.append(a + b)
                else:                   # assume '-'
                    stack.append(a*b)

        if word:                        # flush trailing number if any
            stack.append(int(word))

        assert len(stack) == 1
        return stack[0]

    def __getitem__(self, idx):
        num_words = random.choice([2, 3, 4, 5, 6])
        #print(num_words)
        word_cnt, op_cnt = 0, 0
        seq = []

        # build a valid alternating sequence with spaces between tokens
        while word_cnt <= num_words and op_cnt < num_words - 1:
            if word_cnt - op_cnt < 2:
                # need at least 2 numbers before an op is legal
                word_cnt += 1
                w = self.generate_word()
                seq.extend(list(w))     # add number chars
            elif word_cnt == num_words:
                op_cnt += 1
                op = random.choice(operations)
                seq.append(op)
            elif random.random() < 0.5:
                word_cnt += 1
                w = self.generate_word()
                seq.extend(list(w))
            else:
                op_cnt += 1
                op = random.choice(operations)
                seq.append(op)
            seq.append(' ')
            #print(seq, word_cnt, op_cnt)       # space as separator

        ans = self.evaluate(seq)
        seq.append('=')                 # trailing '=' as you had
        seq.append(' ')
        # Convert to ids
        seq_ids = [stoi[c] for c in seq]
        ans_ids = [stoi[c] for c in str(ans)]
        return torch.tensor(seq_ids, dtype=torch.long), torch.tensor(ans_ids, dtype=torch.long)

num_list = []  

def get_num(num_digits, num): 
  if num_digits == 0: 
    global num_list 
    num_list.append(num) 
  else: 
    for x in digits: 
      get_num(num_digits -1, num + [x]) 

import operator 

class ArithmeticDataset(Dataset): 
  def __init__(self, digits, op=operator.add, symbol = "+"): 
    # Lazy number space: avoid materializing all numbers for large digits
    self.digits = int(digits)
    self.N = 10 ** self.digits
    self.op = op 
    self.symbol = symbol 
  def __len__(self): 
    return self.N*self.N 
  def _idx_to_digits(self, n):
    s = ("%0" + str(self.digits) + "d") % n
    return list(s)
  def __getitem__(self, idx): 
    i, j = divmod(idx, self.N) 
    x = self._idx_to_digits(i)
    y = self._idx_to_digits(j)
    ans = self._apply(x, y) 
    # RPN prompt: x ' ' y ' ' op ' ' '=' ' '
    tokens = x + [" "] + y + [" "] + [self.symbol] + [" "] + ["="] + [" "] 
    token_ids = [stoi[c] for c in tokens]
    ans_ids = [stoi[c] for c in ans]
    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(ans_ids, dtype=torch.long) 
  def _apply(self, x, y): 
    str1 = "".join(x) 
    str2 = "".join(y) 
    res = self.op(int(str1), int(str2)) 
    return list(str(res)) 
    