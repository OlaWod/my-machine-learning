# 注意力,多头注意力,自注意力及Pytorch实现

本文介绍注意力机制（Attention mechanism），多头注意力（Multi-head attention），自注意力（self-attention），以及它们的Pytorch实现。如有错误，还望指出。

关于attention最著名的文章是[Attention Is All You Need](https://arxiv.org/abs/1706.03762v5)，作者提出了Transformer结构，里面用到了attention。

## 一、注意力机制（Attention mechanism）

在[Attention Is All You Need](https://arxiv.org/abs/1706.03762v5) 3.2 节中讲的很清楚了：

> An **attention function** can be described as mapping a **query** and a set of **key-value** pairs to an output, where the query, keys, values, and output are all vectors.
The **output** is computed as **a weighted sum of the values**, where the **weight** assigned to each value is computed by a **compatibility function** of the **query** with the corresponding **key**.

输入是query和 key-value，注意力机制首先计算query与每个key的关联性（compatibility），每个关联性作为每个value的权重（weight），各个权重与value的乘积相加得到输出。

[Attention Is All You Need](https://arxiv.org/abs/1706.03762v5) 中用到的attention叫做“Scaled Dot-Product Attention”，具体过程如下图所示：

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/1.PNG)

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/2.PNG)

具体计算过程如下图所示：

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/3.PNG)

代码实现如下：

```python
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale
        
        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        
        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output
        
        
if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64
    
    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn)
    print(output)
```

## 二、多头注意力（Multi-head attention）

上述只求一次注意力的过程可以叫做单头注意力。多头注意力就是对同样的Q, K, V求多次注意力，得到多个不同的output，再把这些不同的output连接起来得到最终的output。过程如下图所示：

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/4.PNG)

计算过程如下图所示：

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/5.PNG)

代码实现如下：

```python
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output
        
        
if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    
    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)    
    mask = torch.zeros(batch, n_q, n_k).bool()
    
    mha = MultiHeadAttention(n_head=8, d_k_=128, d_v_=64, d_k=256, d_v=128, d_o=128)
    attn, output = mha(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())
```

## 三、自注意力（self-attention）

当注意力的query和key、value全部来自于同一个东西时，就称为自注意力。如下图所示，query和key、value全都来源于X。

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/6.PNG)

自注意力通过X求query和key、value的计算过程如下图所示：

![](https://github.com/OlaWod/my-machine-learning/blob/master/Attention/figs/7.PNG)

代码实现如下：

```python
class SelfAttention(nn.Module):
    """ Self-Attention """
    
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)   
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output
        
        
if __name__ == "__main__":
	n_x = 4
    d_x = 80
    
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()
    
    selfattn = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=80, d_o=80)
    attn, output = selfattn(x, mask=mask)

    print(attn.size())
    print(output.size())
```

## 四、Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlaWod/my-machine-learning/blob/master/Attention/attention.ipynb)