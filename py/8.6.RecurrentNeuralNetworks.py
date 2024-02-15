# 1. 预处理
# source: 8.2.1. 读取数据集
import os, hashlib, requests, re

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# source: 8.2.2. 词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
        
# source: 8.2.3. 词表
import collections
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break # 前面有排序, 词频从大到小排序, 也就是当出现第一个词频小于min_freq时, 后面的全部会小于min_freq
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def count_corpus(self, tokens):
        """统计词元的频率"""
        # 这里的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens) 

# --------------------------------------------------------------------------- #
# 2. 构建数据迭代器和词表
# source: 8.2.4. 整合所有功能
import random, torch
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()      # 读取数据集
    tokens = tokenize(lines, 'char') # 词元化
    vocab = Vocab(tokens)            # 词表
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    
    """
    corpus2 = []
    for index, line in enumerate(tokens):
        for token in line:
            corpus2.append(vocab[token])
    assert(corpus == corpus2)
    """
    
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# source: 8.3.4.1. 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# source: 8.3.4.2. 顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random     # 随机采样
        else:
            self.data_iter_fn = seq_data_iter_sequential # 顺序分区
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps) # train_iter: feature-label迭代器; vocab: 词表
# --------------------------------------------------------------------------- #
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# --------------------------------------------------------------------------- #
# 3. 定义模型
from torch import nn

# # source: 8.5.2. 初始化模型参数
# def get_params(vocab_size, num_hiddens, device):
#     num_inputs = num_outputs = vocab_size

#     def normal(shape):
#         return torch.randn(size=shape, device=device) * 0.01

#     # 隐藏层参数
#     W_xh = normal((num_inputs, num_hiddens))      # 输入维度(d), 隐藏单元的数量(h)
#     W_hh = normal((num_hiddens, num_hiddens))     # 隐藏单元的数量(h), 隐藏单元的数量(h)
#     b_h = torch.zeros(num_hiddens, device=device) # 隐藏单元的数量(h)
#     # 输出层参数
#     W_hq = normal((num_hiddens, num_outputs))     # 隐藏单元的数量(h), 输出维度(q)
#     b_q = torch.zeros(num_outputs, device=device) # 输出维度(q)
#     # 附加梯度
#     params = [W_xh, W_hh, b_h, W_hq, b_q]
#     for param in params:
#         param.requires_grad_(True)
#     return params

# # source: 8.5.3 循环神经网络模型
# def init_rnn_state(batch_size, num_hiddens, device):
#     return (torch.zeros((batch_size, num_hiddens), device=device), )

# def rnn(inputs, state, params):
#     # inputs的形状: (时间步数量, 批量大小, 词表大小)
#     W_xh, W_hh, b_h, W_hq, b_q = params
#     H, = state
#     outputs = []
#     # X的形状: (批量大小, 词表大小)
#     for X in inputs:
#         H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h) # eq.8.4.5
#         # Y的形状: (批量大小, 词表大小)
#         Y = torch.mm(H, W_hq) + b_q                                 # eq.8.4.6
#         outputs.append(Y)
#     # torch.cat(outputs, dim=0)形状: (时间步数 × 批量大小, 词表大小)
#     # 隐状态H形状: (批量大小, 隐藏单元数)
#     return torch.cat(outputs, dim=0), (H,)

# class RNNModelScratch:
#     """从零开始实现的循环神经网络模型"""
#     def __init__(self, vocab_size, num_hiddens, device,
#                  get_params, init_state, forward_fn):
#         self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
#         self.params = get_params(vocab_size, num_hiddens, device)
#         self.init_state, self.forward_fn = init_state, forward_fn

#     def __call__(self, X, state):                                           # 输入X的形状: (批量大小, 时间步数量)
#         X = nn.functional.one_hot(X.T, self.vocab_size).type(torch.float32) # 此时, X的形状: (时间步数量, 批量大小, 词表大小)
#         return self.forward_fn(X, state, self.params)                       # 执行 rnn()

#     def begin_state(self, batch_size, device):
#         return self.init_state(batch_size, self.num_hiddens, device)

# num_hiddens = 512
# net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params, init_rnn_state, rnn)
# --------------------- #
# source: 8.6.1. 定义模型
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state): # 相当于 class RNNModelScratch 类中的 __call__()
        X = nn.functional.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
net = RNNModel(rnn_layer, vocab_size=len(vocab))
# --------------------------------------------------------------------------- #
# 4.预测
# source: 8.5.4. 预测
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state) # 更新隐状态
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
# --------------------------------------------------------------------------- #
# 5. 梯度裁剪
# source: 8.5.5. 梯度裁剪
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum( torch.sum((p.grad ** 2)) for p in params ))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
# --------------------------------------------------------------------------- #
# 6.训练
# source: 8.5.6. 训练
import time, numpy, math
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return numpy.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        # 隐状态
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        
        # y shape = (时间步数 × 批量大小)
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        # X shape = (批量大小/batch_size, 时间步数量/num_steps)
        # state[0] shape = (批量大小, 隐藏单元数)
        # y_hat shape = (时间步数 × 批量大小, 词表大小)
        y_hat, state = net(X, state)     # 神经网络
        
        l = loss(y_hat, y.long()).mean() # 损失函数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()                 # 反向传播
            grad_clipping(net, 1)        # 梯度裁剪
            updater.step()               # 优化
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    # metric[0]: 累积训练损失之和
    # metric[1]: 累积已训练词元数量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # if (epoch + 1) % 10 == 0:
        #     print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# main
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu())