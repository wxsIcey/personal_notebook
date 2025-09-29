import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 批处理维度上的元素是独立的，永远不会相互通信！！
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril是一个下三角矩阵，用于在计算注意力分数时屏蔽掉未来的信息
        # 这样可以确保模型在预测下一个token时，只能看到当前和之前的token
        # tril[:T, :T]会生成一个形状为(T,T)的下三角矩阵
        # 其中T是输入序列的长度，这个矩阵的形状与注意力分数矩阵相匹配
        # 这样可以确保在计算注意力分数时，只考虑当前和之前的token
        # 例如，如果block_size为32，那么tril的形状为(32, 32)
        # tril的下三角部分为1，上三角部分为0
        # 这样在计算注意力分数时，可以通过将tril与注意力分数矩阵相乘来屏蔽掉未来的信息   
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # dropout层，用于在训练时随机丢弃一些注意力分数，以防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        
        # q和k之间的点积计算注意力分数，（B, T, T)表示每个时间步对其他时间步的相关性
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # 使用下三角矩阵tril，将未来时间步的注意力分数设置为负无穷，确保当前时间步只能关注当前及之前的时间步
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        # 使用注意力权重对值向量进行加权求和，得到每个时间步的上下文表示
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # 创建一个包含num_heads个Head的ModuleList，每个Head的大小为head_size
        # 每个Head独立计算注意力分数
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 用于将所有注意力头的输出拼接起来的线性层
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 遍历 self.heads 中的每个 Head，将输入 x 传递给每个注意力头。
        # 每个 Head 的输出形状为 (B, T, head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# 该部分是简单的MLP的实现，包含线性层和非线性激活函数，通常用于Transformer的前馈网络部分，增强模型的表达能力
# FFN(Feed Forward Neural Network)前馈神经网络 在Transformer的上下文中，是一种特殊设计和放置的MLP
# 它位于每个Transformer块的注意力机制之后，负责对每个位置的表示进行独立的非线性变换
# 这种设计使得模型能够捕捉复杂的模式和特征，增强其表达能力
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        # 定义一个名为self.net的序列模型，包含两个线性层和一个ReLU激活函数
        self.net = nn.Sequential(
            # 线性层，将输入维度从n_embed映射到4倍的n_embd(升维操作，增加特征的表达能力)
            nn.Linear(n_embd, 4 * n_embd), #（在更高维的空间中，对信息进行更精细、更复杂的非线性变换，从而极大地增强模型的表示能力）
            # ReLU激活函数，增加非线性，将负值变为0，正值保持不变
            nn.ReLU(),
            # 线性层，将4倍的n_embd映射回n_embd(降维操作，恢复到原始的嵌入维度)
            nn.Linear(4 * n_embd, n_embd),
            # Dropout层，随机丢弃一定比例的神经元，防止过拟合
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# 这个部分就是Transformer的基本块（Block）的实现，包含了多头自注意力机制和前馈网络（整个图的核心部分！！！）
# 每个Block包含一个多头自注意力层和一个前馈网络层，
# 以及两个层归一化（Layer Normalization）操作，通常用于Transformer模型
# 通过残差连接（skip connection）将输入和输出相加，增强模型的稳定性和训练效果
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 输入的嵌入向量维度为n_embd，通过线性变化分割为n_head个头，每个头的维度为head_size
# 每个头独立计算注意力分数：
#     注意力权重：softmax(Q @ K^T / sqrt(head_size))
#     加权值：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_size)) @ V
# 将所有头的输出拼接起来，恢复到原始的嵌入维度 n_embd
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # 输入（B,T）的正数向量，B为batch size,T为序列长度，输入序列中token在词汇表中的索引范围为[0, vocab_size-1]
        # 比如样本为16个，每个样本包含的token数量为32，那么输入的idx形状为（16,32）
        # 输出形状为（B,T,n_embd），其中n_embd是嵌入向量的维度
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # 线性层，输入维度为n_embd，输出维度为vocab_size，也就是将每个嵌入向量从n_embd维度映射到vocab_size维度
        # 输出的每个维度对应词汇表中的一个token，表示该token的预测分数
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # 输入x一次通过self.blocks的每个Block，每个Block对输入进行处理（注意每个Block的输入和输出形状保持一致）
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        # 模型对每个token的预测倾向，模型根据输入序列，预测下一个token是词汇表中每个可能token的概率分布或分数
        logits = self.lm_head(x) # (B,T,vocab_size)
        # 假设B为2，T为3，vocab_size为5，那么logits的形状为(2, 3, 5)，表示2个样本，每个样本有3个时间步，每个时间步有5个token的预测分数
        # logits = torch.tensor([
        #     [  # 第一个样本 (B=0)
        #         [2.0, 1.0, 0.1, -1.0, 0.5],  # 第一个时间步 (T=0)
        #         [1.5, 0.2, -0.5, 0.0, 1.0],  # 第二个时间步 (T=1)
        #         [0.0, 2.0, 1.0, -0.5, 0.5],  # 第三个时间步 (T=2)
        #     ],
        #     [  # 第二个样本 (B=1)
        #         [1.0, 0.5, -0.5, 2.0, 0.0],  # 第一个时间步 (T=0)
        #         [0.0, 1.0, 2.0, -1.0, 0.5],  # 第二个时间步 (T=1)
        #         [1.5, 0.0, -0.5, 0.2, 1.0],  # 第三个时间步 (T=2)
        #     ]
        # ])  # logits 的形状为 (2, 3, 5)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))