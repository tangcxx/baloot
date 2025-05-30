## 作弊的单一 sun 模型


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
        self.fc1 = nn.Linear((4+nth)*32, 512)
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

        self.fc5 = nn.Linear(512, 32)

    def forward(self, x):
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

        return self.fc5(x)

# %%
from _game import Game

class Bot:
    def __init__(self, models):
        self.models = models

    def reset(self, game: Game):
        self.game = game

        self.data = []

    def play(self, game: Game):
        choices = game.get_choices()
        if len(choices) == 1:
            return choices[0]

        round = len(game.history) - 1
        order = len(game.history[-1]["played"])
        seat = (game.host + order) % 4
        model = self.models[order]

        x = self.get_data(game)
        logits = model(x)
        mask = get_mask(choices)
        masked_logits = torch.where(mask, logits, torch.tensor(float("-inf")))

        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        self.data.append((round, order, log_prob, seat))
        
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

        x = np.zeros((4+order, 32), np.int8)

        for i in range(4):
            x[i, 0:32] = game.cards_vec[seats[i]]  ## 手牌, 从自己开始
        for i, card in enumerate(played):
            x[4+i, card] = 1  ## 出过的牌, 按本轮出牌顺序

        x = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        
        return x
        
#%%
