## _bot_f 衍生
## 增加 baseline

#%%
import numpy as np 
import torch
from torch import nn

from _game_torch import Game

#%%
def get_mask(choices, n_actions=32):
    mask = torch.zeros(n_actions)
    mask[choices] = 1
    return mask.bool().unsqueeze(0)

# %%
class Model(nn.Module):
    def __init__(self, nth):
        super().__init__()
        self.emb = nn.Embedding(32, 5)
        
        self.lstm = nn.LSTM(5*7, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128 + (9 + nth) * 5 + 16, 512)
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
    
    # def embsum(self, cards_list):
    #     if len(cards_list) == 0:
    #         return torch.zeros(5, dtype=torch.float32)
    #     return self.emb(torch.tensor(cards_list, dtype=torch.int64)).sum(dim=-2)
    
    def encoder(self, host, order, card_revealed, cards, cards_played, outof, history):
        played = history[-1]["played"]
        seat = (host + order) % 4
        seats = [(seat + i) % 4 for i in range(4)]  # 从自己开始，各家的座位序号
        revealed_emb = torch.zeros((4, 5), dtype=torch.float32)
        revealed_emb[-seat % 4] = self.emb(torch.tensor(card_revealed, dtype=torch.int64))
        hands_emb = self.emb(torch.tensor(cards[seat], dtype=torch.int64)).sum(dim=-2).unsqueeze(0)
        next_1_played_emb = self.emb(torch.tensor(cards_played[seats[1]], dtype=torch.int64)).sum(dim=-2).unsqueeze(0)
        next_2_played_emb = self.emb(torch.tensor(cards_played[seats[2]], dtype=torch.int64)).sum(dim=-2).unsqueeze(0)
        next_3_played_emb = self.emb(torch.tensor(cards_played[seats[3]], dtype=torch.int64)).sum(dim=-2).unsqueeze(0)
        remains_emb = self.emb(torch.tensor(cards[seats[1]] + cards[seats[2]] + cards[seats[3]], dtype=torch.int64)).sum(dim=-2).unsqueeze(0)
        played_emb = self.emb(torch.tensor(played, dtype=torch.int64))
        x = torch.cat([revealed_emb, hands_emb, next_1_played_emb, next_2_played_emb, next_3_played_emb, remains_emb, played_emb], dim=0).reshape(1, -1)
        x = torch.cat([x, outof[seats].reshape(1, 16)], dim=-1)
        
        h = torch.zeros((6,7), dtype=torch.int64)
        for i in range(6):
            if i >= len(history):
                break
            host, played = history[i]["host"], history[i]["played"]
            for j, card in enumerate(played):
                h[i, (host + j) % 4] = card  
        z = self.emb(h).reshape(1, 6, 35)

        return z, x

    def forward(self, host, order, card_revealed, cards, cards_played, outof, history):
        z, x = self.encoder(host, order, card_revealed, cards, cards_played, outof, history)  # z is of shape (batch_size, 6, 7, 32)
        
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

        logits, v = model(*self.get_data(game))
        mask = get_mask(choices)
        masked_logits = torch.where(mask, logits, torch.tensor(float("-inf")))

        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        self.data.append((round, order, log_prob, seat, v.squeeze()))
        
        # print("in Bot", round, order, seat, choices, log_prob)
        return action.item()

## 输入信息
## 明牌 
## 四个人的手牌（按本轮顺位排序）
## 本轮在我之前的玩家的出牌 32 * 3

    def get_data(self, game: Game):
        
        played = game.history[-1]["played"]
        order = len(played)
        seats = [(game.host + order + i) % 4 for i in range(4)] ## 从自己开始，各家的座位序号

        return game.host, order, game.card_revealed, game.cards, game.cards_played, game.outof, game.history

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
        with torch.no_grad():
            logits, _ = model(*self.get_data(game))
        return choices[logits[0, choices].argmax()]
        

#%%
