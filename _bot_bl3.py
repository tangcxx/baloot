## _bot_bl 衍生
## 断牌数据的排列 game.outof -> game.outof[seats]

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
        
        self.fc1 = nn.Linear(128 + 32 * (9 + nth) + 16, 512)
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

        self.p = nn.Linear(512, 32)
        
        self.v = nn.Linear(512, 1)  # baseline value function

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

        return self.p(x), self.v(x)  # return both policy logits and value function


# %%
from _game import Game

class Bot:
    def __init__(self, models):
        self.models = models

    def reset(self, game: Game):
        self.game = game
        self.data = []
        for i in range(4):
            self.models[i].train()

    def play(self, game: Game, choices):
        # choices = game.get_choices()
        if len(choices) == 1:
            return choices[0]

        round = len(game.history) - 1
        order = len(game.history[-1]["played"])
        seat = (game.host + order) % 4
        model = self.models[order]

        z, x = self.get_data(game)
        logits, v = model(z, x)
        mask = get_mask(choices)
        masked_logits = torch.where(mask, logits, torch.tensor(float("-inf")))

        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        self.data.append((round, order, log_prob, seat, v.squeeze()))
        
        # print("in Bot", round, order, seat, choices, log_prob)
        return action

## 输入信息
## 明牌 
## 四个人的手牌（按本轮顺位排序）
## 本轮在我之前的玩家的出牌 32 * 3

    def get_data(self, game: Game):
        played = game.history[-1]["played"]
        order = len(played)
        seats = [(game.host + order + i) % 4 for i in range(4)] ## 从自己开始，各家的座位序号

        x = np.zeros((9+order, 32), dtype=np.int8)

        ## 明牌怎么编码？用4*32位编码1张明牌？
        x[(-game.host-order) % 4, game.card_revealed] = 1  ## 明牌
        
        x[4, 0:32] = game.cards_vec[seats[0]]  ## 自己手牌
        x[5, 0:32] = game.cards_vec_init[seats[1]] - game.cards_vec[seats[1]] ## 下一位出过的牌
        x[6, 0:32] = game.cards_vec_init[seats[2]] - game.cards_vec[seats[2]] ## 再下一位出过的牌
        x[7, 0:32] = game.cards_vec_init[seats[3]] - game.cards_vec[seats[3]] ## 再下一位出过的牌
        
        ## 其他各家余牌
        x[8, 0:32] = game.cards_vec[seats[1]] + game.cards_vec[seats[2]] + game.cards_vec[seats[3]]
        
        ## 本轮在我之前的各家出牌
        for i, card in enumerate(played):
            x[9+i, card] = 1
        
        ## 各家暴露的断牌
        x = np.concat([x.reshape(-1), game.outof[seats].reshape(-1)])

        z = np.zeros((6, 7, 32))
        
        for i in range(min(6, len(game.history) - 1)):
            h = game.history[i]
            host, played = h["host"], h["played"]
            for j, card in enumerate(played):
                z[i, host + j, card] = 1

        z = torch.tensor(z, dtype=torch.float32).reshape(1, 6, 7 * 32)
        x = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        
        return z, x
        
        
class Bot_Eval(Bot):
    def __init__(self, models):
        self.models = models

    def reset(self, game: Game):
        self.game = game
        for i in range(4):
            self.models[i].eval()
    
    def play(self, game: Game, choices):
        # choices = game.get_choices()
        if len(choices) == 1:
            return choices[0]
        order = len(game.history[-1]["played"])
        model = self.models[order]
        z, x = self.get_data(game)
        with torch.no_grad():
            logits, _ = model(z, x)
        return choices[logits[0, choices].argmax()]
        

#%%
