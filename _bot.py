## 定义Bot类
## reset(): 重置bot状态
## play(): 出牌
## get_xs(): 获取当前状态

## 定义模型
## 使用策略梯度法
## 对于baloot，作弊可能很容易被察觉。先训练不作弊的模型。
## 按照一轮中的出牌次序，使用四个不同的模型
## 输入信息
## 明牌 32 
## 我的手牌 32
## 各家已经出过的牌 32 * 3
## 场上其他各家余牌 32
## 本轮在我之前的玩家的出牌 32 * n

## 是否利用花色的对称性扩充样本量？

## 之前的出牌怎么编码？
## 一轮的数据编码为 7*32
## 第一个出牌者的出牌数据放在 [seat * 32 : (seat+1)*32]，后面的依此类推，其他位置为0

## 奖励应该如何定义？
## 以一局的胜负作为奖励似乎不妥。以本轮至终局的得分作为奖励。
## 用什么作为critic? 先不考虑
## 己方得分，正奖励，对方得分，负奖励

#%%
import numpy as np 
import torch
from torch import nn

# %%
class Model(nn.Module):
    def __init__(self, nth):
        super().__init__()
        self.lstm = nn.LSTM(224, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128 + 32*(6+nth), 512)
        self.norm1 = nn.LayerNorm(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 512)
        self.norm3 = nn.LayerNorm(512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 512)
        self.norm4 = nn.LayerNorm(512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(512, 512)
        self.norm5 = nn.LayerNorm(512)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(512, 32)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        h_n = h_n.squeeze(0)
        x = torch.cat([h_n,x], dim=-1)
        
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        return self.fc6(x)

# %%
from _game import Game

class Bot:
    def __init__(self, models):
        self.models = models
        pass

    def reset(self, game: Game):
        self.game = game
        self.data = []

    def play(self, game: Game):
        choices = game.get_choices()
        if len(game.history) <= 7:
            order = len(game.history[-1]["played"])
            seat = (game.host + order) % 4
            z, x = self.get_data(game)
            logits = self.models[order](z, x)
            mask = torch.full_like(logits, fill_value=float('-inf'))

            mask[0, choices] = 0.0

            masked_logits = logits + mask  
            action_dist = torch.distributions.Categorical(logits=masked_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            self.data.append((order, log_prob, seat))
        else:
            action = choices[0]
        return action

## 定义模型
## 使用策略梯度法
## 对于baloot，作弊可能很容易被察觉。先训练不作弊的模型。
## 按照一轮中的出牌次序，使用四个不同的模型
## 输入信息
## 明牌 40 
## 我的手牌 40
## 各家已经出过的牌 40 * 3
## 场上其他各家余牌 40
## 本轮在我之前的玩家的出牌 40 * n
## 双方分差，场上剩余分数，各花色剩余分数
## 40 是因为主牌单独编码，即4种花色+主牌各8张
####  如果没有主牌，取前32项，如果有主牌，对应的花色的8项置0，主牌数据放在最后8项。
## 输入数据长度240 + 40*n
## 是否利用花色的对称性扩充样本量？

## 之前的出牌怎么编码？
## 一轮的数据编码为 7*40
## 第一个出牌者的出牌数据放在 [seat * 40 : (seat+1)*40]，后面的依此类推，其他位置为0    
    def get_data(self, game: Game):
        played = game.history[-1]["played"]
        seat_offset = len(played)
        seat = (game.host + seat_offset) % 4
        seats = [(seat + i) % 4 for i in range(4)] ## 从自己开始，各家的座位序号

        x = np.zeros((6+seat_offset, 32))

        x[0, game.card_revealed] = 1  ## 明牌
        x[1, 0:32] = game.cards_vec[seat]  ## 手牌
        x[2, 0:32] = game.cards_vec_init[seats[1]] - game.cards_vec[seats[1]] ## 下一位出过的牌
        x[3, 0:32] = game.cards_vec_init[seats[2]] - game.cards_vec[seats[2]] ## 再下一位出过的牌
        x[4, 0:32] = game.cards_vec_init[seats[3]] - game.cards_vec[seats[3]] ## 再上一位出过的牌
        
        ## 其他各家余牌
        x[5, 0:32] = game.cards_vec[seats[1]] + game.cards_vec[seats[2]] + game.cards_vec[seats[3]]
        
        ## 本轮在我之前的各家出牌
        for i, card in enumerate(played):
            x[6+i, card] = 1

        z = np.zeros((6, 7, 32))
        
        for i in range(6):
            if i >= len(game.history):
                break
            h = game.history[i]
            host, played = h["host"], h["played"]
            for j, card in enumerate(played):
                z[i, host + j, card] = 1

        z = torch.tensor(z, dtype=torch.float32).reshape(1, 6, 7 * 32)
        x = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        
        return z, x
        
#%%
p = torch.tensor([1,2,7], dtype=torch.float32)