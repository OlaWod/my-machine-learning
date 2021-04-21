import torch
import torch.nn as nn
import numpy as np


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


class SimpleSelfAttention(nn.Module):
    """ Simple Self-Attention """

    def __init__(self, n_head, d_model):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.fc_q = nn.Linear(d_model, n_head * self.d_k)
        self.fc_k = nn.Linear(d_model, n_head * self.d_k)
        self.fc_v = nn.Linear(d_model, n_head * self.d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(self.d_k, 0.5))

        self.fc_o = nn.Linear(n_head * self.d_v, d_model)

    def forward(self, x, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_x, d_model = x.size()
        n_q = n_k = n_v = n_x

        q = self.fc_q(x) # 1.单头变多头
        k = self.fc_k(x)
        v = self.fc_v(x)
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
    batch = 5
    
    # SHA
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64
    
    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)    
    mask = torch.zeros(batch, n_q, n_k).bool()

    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())

    # MHA
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

    # self-attention
    n_x = 4
    d_x = 80
    
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()
    
    selfattn = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=80, d_o=80)
    attn, output = selfattn(x, mask=mask)

    print(attn.size())
    print(output.size())

    # self-attention
    n_x = 4
    d_x = 80
    
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()
    
    mha = MultiHeadAttention(n_head=8, d_k_=80, d_v_=80, d_k=128, d_v=64, d_o=80)
    attn, output = mha(x, x, x, mask=mask)

    print(attn.size())
    print(output.size())

    # self-attention
    n_x = 4
    d_x = 81
    
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x).bool() # (batch, n_k)
    mask = mask.unsqueeze(1).expand(-1, n_x, -1) # (batch, n_x, n_x), (batch, n_q, n_k)

    selfattn = SimpleSelfAttention(n_head=8, d_model=d_x)
    attn, output = selfattn(x, mask=mask)

    print(attn.size())
    print(output.size())
