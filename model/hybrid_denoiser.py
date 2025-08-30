import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp


class hybridST(nn.Module):
    def __init__(self, layers, d_hid):
        super().__init__()
        self.stcformer = hybridSTFormer(layers, d_hid)
        self.regress_head = nn.Sequential(
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, 1)
        )

    def forward(self, x):
        x = self.stcformer(x)
        x = self.regress_head(x).squeeze(-1)
        x = x.mean(-1)

        return x


class hybridSTFormer(nn.Module):
    def __init__(self, num_block, d_coor ):
        super(hybridSTFormer, self).__init__()

        self.num_block = num_block
        self.d_coor = d_coor
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1,1,63,d_coor))
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1,300,1,d_coor))

        self.stc_block = []
        for l in range(self.num_block):
            self.stc_block.append(hybridST_BLOCK(self.d_coor))
        self.stc_block = nn.ModuleList(self.stc_block)

    def forward(self, input):
        input = input + self.spatial_pos_embedding + self.temporal_pos_embedding
        for i in range(self.num_block):
            input = self.stc_block[i](input)

        return input


class hybridST_BLOCK(nn.Module):
    def __init__(self, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)

        self.mlp = Mlp(d_coor, d_coor, d_coor)

        self.stc_att = hybridST_ATTENTION( d_coor)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.stc_att(input)
        x = x + self.mlp(self.layer_norm(x))

        return x


class hybridST_ATTENTION(nn.Module):
    def __init__(self,d_coor, head=8):
        super().__init__()
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)

        self.scale = (d_coor // 2) ** -0.5
        self.proj = nn.Linear(d_coor, d_coor)
        self.head = head

        self.emb = nn.Embedding(5, d_coor//head//2)
        self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()

        self.sep2_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.sep2_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)


    def forward(self, input):
        b, t, s, c = input.shape

        h = input
        x = self.layer_norm(input)

        qkv = self.qkv(x)  
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  
       
        qkv_s, qkv_t = qkv.chunk(2, 4) 

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2] 

        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head) 
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head) 

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head) 
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  

        att_s = (q_s @ k_s) * self.scale 
        att_t = (q_t @ k_t) * self.scale 

        att_s = att_s.softmax(-1) 
        att_t = att_t.softmax(-1)  

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        v_s = rearrange(v_s, 'b (h c) t s   -> (b h t) s c ', h=self.head)  
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head) 

        x_s = att_s @ v_s 
        x_t = att_t @ v_t 

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)  
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  

        x_t = x_t

        x = torch.cat((x_s, x_t), -1) 
        x = rearrange(x, 'b h t s c -> b  t s (h c) ')

        x = self.proj(x)
        x = x + h
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


if __name__ == "__main__":
    pass

