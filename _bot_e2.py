## DeepSets: 一组牌 embedding 之后, 增加一层 MLP , 然后再求和

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
        
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.lstm = nn.LSTM(5*7, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128 + 20 + 5 * 32 + nth * 5 + 16, 512)
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
    
    def encoder(self, revealed_owner, card_revealed, hands, next_1_played, next_2_played, next_3_played, remains, played, outof, h):
        revealed = torch.zeros((4, 5), dtype=torch.float32)
        revealed[revealed_owner] = self.emb(card_revealed)
        revealed = revealed.reshape(1, -1)  ## shape: (1, 20)
        hands = self.mlp(self.emb(hands)).sum(dim=-2).unsqueeze(0)  ## shape: (1, 32)
        next_1_played = self.mlp(self.emb(next_1_played)).sum(dim=-2).unsqueeze(0) ## shape: (1, 32)
        next_2_played = self.mlp(self.emb(next_2_played)).sum(dim=-2).unsqueeze(0) ## shape: (1, 32)
        next_3_played = self.mlp(self.emb(next_3_played)).sum(dim=-2).unsqueeze(0) ## shape: (1, 32)
        remains = self.mlp(self.emb(remains)).sum(dim=-2).unsqueeze(0) ## shape: (1, 32)
        played = self.emb(played).reshape(1, -1)
        
        x = torch.cat([revealed, hands, next_1_played, next_2_played, next_3_played, remains, played, outof], dim=-1)
        
        z = self.emb(h).reshape(1, 6, 35)

        return z, x

    def forward(self, revealed_owner, card_revealed, hands, next_1_played, next_2_played, next_3_played, remains, played, outof, h):
        z, x = self.encoder(revealed_owner, card_revealed, hands, next_1_played, next_2_played, next_3_played, remains, played, outof, h)  # z is of shape (batch_size, 6, 7, 32)
        
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
        history = game.history
        played_the_round = history[-1]["played"]
        order = len(played_the_round)
        seat = (game.host + order) % 4
        seats = [(game.host + order + i) % 4 for i in range(4)] ## 从自己开始，各家的座位序号
        cards, cards_played = game.cards, game.cards_played

        revealed_owner = (game.revealed_owner - seats[0]) % 4
        card_revealed = torch.tensor(game.card_revealed, dtype=torch.int64)
        hands = torch.tensor(cards[seat], dtype=torch.int64)
        next_1_played = torch.tensor(cards_played[seats[1]], dtype=torch.int64)
        next_2_played = torch.tensor(cards_played[seats[2]], dtype=torch.int64)
        next_3_played = torch.tensor(cards_played[seats[3]], dtype=torch.int64)
        remains = torch.tensor(cards[seats[1]] + cards[seats[2]] + cards[seats[3]], dtype=torch.int64)
        played_the_round = torch.tensor(played_the_round, dtype=torch.int64)
        outof = game.outof[seats].reshape(1, -1) ## shape: (1, 16)
        
        h = torch.zeros((6,7), dtype=torch.int64)
        for i in range(6):
            if i >= len(history):
                break
            host, played = history[i]["host"], history[i]["played"]
            for j, card in enumerate(played):
                h[i, (host + j) % 4] = card  

        return revealed_owner, card_revealed, hands, next_1_played, next_2_played, next_3_played, remains, played_the_round, outof, h

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
