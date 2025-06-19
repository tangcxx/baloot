## 价值网络

#%%
import numpy as np 
import torch
from torch import nn

#%%
def get_mask(choices, n_actions=32):
    mask = torch.zeros(n_actions)
    mask[choices] = 1
    return mask.bool().unsqueeze(0)

# %%
class Model(nn.Module):
    def __init__(self, nth):
        super().__init__()
        self.lstm = nn.LSTM(32*7, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128 + 32 * (6 + nth + 1) + 16 + 4, 512)
        self.norm1 = nn.LayerNorm(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.v = nn.Linear(256, 1)  # baseline value function

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

        return self.v(x)  # return both policy logits and value function


# %%
from _game2 import Game

class Bot:
    def __init__(self, models, eps=0):
        self.models = models
        self.eps = eps  # exploration rate

    def reset(self, game: Game):
        self.game = game
        self.data = []

    def play(self, game: Game, choices):
        round = len(game.history) - 1  ## 从0开始数
        order = len(game.history[-1]["played"])
        z, x = self.get_data(game, choices)

        if len(choices) == 1:
            choice_idx = 0
        
        elif self.eps > 0 and torch.rand(1).item() < self.eps:  # epsilon-greedy exploration
            choice_idx = np.random.choice(len(choices))
        
        else:
            model = self.models[order]
            model.eval()
            with torch.no_grad():
                values = model(z, x)
            choice_idx = values.argmax()

        self.data.append((round, order, z[choice_idx:choice_idx+1], x[choice_idx:choice_idx+1]))
        return choices[choice_idx]

## 输入信息
## 明牌 
## 四个人的手牌（按本轮顺位排序）
## 本轮在我之前的玩家的出牌 32 * 3

    def get_data(self, game: Game, choices):
        n_choices = len(choices)
        played = game.history[-1]["played"]
        order = len(played)
        seats = [(game.host + order + i) % 4 for i in range(4)] ## 从自己开始，各家的座位序号

        x = np.zeros((7+order, 32), dtype=np.int8)

        ## 明牌怎么编码？用4*32位编码1张明牌？
        revealed_pos = np.zeros(4, dtype=np.int8)
        if np.sum(game.cards_vec, axis=0)[game.card_revealed]:  ## 明牌还没出
            revealed_pos[(game.revealed_owner-game.host-order) % 4] = 1
            x[0, game.card_revealed] = 1  ## 明牌
        
        x[1, 0:32] = game.cards_vec[seats[0]]  ## 自己手牌
        x[2, 0:32] = game.cards_vec_init[seats[1]] - game.cards_vec[seats[1]] ## 下一位出过的牌
        x[3, 0:32] = game.cards_vec_init[seats[2]] - game.cards_vec[seats[2]] ## 再下一位出过的牌
        x[4, 0:32] = game.cards_vec_init[seats[3]] - game.cards_vec[seats[3]] ## 再下一位出过的牌
        
        ## 其他各家余牌
        x[5, 0:32] = game.cards_vec[seats[1]] + game.cards_vec[seats[2]] + game.cards_vec[seats[3]]
        
        ## 本轮在我之前的各家出牌
        for i, card in enumerate(played):
            x[6+i, card] = 1
            
        seq_len = len(game.history)
        z = np.zeros((8, 7, 32))
        
        for i in range(seq_len):
            h = game.history[i]
            host, played = h["host"], h["played"]
            for j, card in enumerate(played):
                z[i, host + j, card] = 1
        z = z.reshape(8, -1)
        
        x = x[np.newaxis, :].repeat(n_choices, axis=0)
        z = z[np.newaxis, :].repeat(n_choices, axis=0)

        for i, choice in enumerate(choices):
            x[i, 6 + order, choice] = 1
        
        x = x.reshape(n_choices, -1)
        outof = game.outof[seats].reshape(-1)
        outof = np.repeat(outof[np.newaxis, :], n_choices, axis=0)
        revealed_pos = np.repeat(revealed_pos[np.newaxis, :], n_choices, axis=0)
        
        ## 各家暴露的断牌, 明牌位置
        x = np.concat([x, outof, revealed_pos], axis=-1)

        x = torch.tensor(x, dtype=torch.float32).reshape(n_choices, -1)
        z = torch.tensor(z, dtype=torch.float32)
        
        return z, x
        
        
class Bot_Eval(Bot):
    def __init__(self, models):
        self.models = models
        self.eps = 0


#%%
