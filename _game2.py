
## 修改断牌的计算，出牌后立刻更新，不再等本轮结束
## _bot_v 时改动
## bl6 及之前的bot用的是 _game.py

#%%
import numpy as np

#%%

COLORS = ['♠', '♥', '♣', '♦']
LABELS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']

cardstr = [c+n 
           for c in COLORS
           for n in LABELS]

def list2vec(l):
    vec = np.zeros(32, dtype=np.int8)
    vec[l] = 1
    return vec

def vec2list(vec):
    return np.where(vec == 1)[0].tolist()

def list2str(l):
    return " ".join([cardstr[i] for i in l])

def vec2str(vec):
    return " ".join([cardstr[i] for i in vec2list(vec)])

## 从7到A的排序
ranks_hokum = [0, 1, 6, 4, 7, 2, 3, 5]
ranks_sun = [0, 1, 2, 6, 3, 4, 5, 7]

## 从7到A的分数
scores_hokum = [0, 0, 14, 10, 20, 3, 4, 11]
scores_sun = [0, 0, 0, 10, 2, 3, 4, 11]

#%%
# class PreGame:
#     def __init__(self):
#         self.cards_shuffled = np.array(list(range(32)), dtype=np.int8) ## 洗牌
#         np.random.shuffle(self.cards_shuffled)
#         self.cards = []
#         self.card_revealed = self.cards_shuffled[20]  ## 第二十一张牌是明牌
        
#     ## 叫牌机器人
#     def register(self, bots_bidding):
#         self.bots_bidding = bots_bidding
#         # [bot.register(self) for bot in self.bots_bidding]
        
#     ## 各发5张牌
#     def deal_befor_bidding(self):
#         cards = self.cards_shuffled[0:20].reshape(4, 5)
#         self.cards_vec = np.array([list2vec(l) for l in cards])
        
#     ## 叫牌
#     def bidding(self):
#         ## 0, 1, 2, 3 对应黑桃，红桃，梅花，方块，4 对应 sun
#         self.hokum = np.random.choice([0, 1, 2, 3, 4])
#         self.host = np.random.choice(4)

#     ## 发剩下的牌
#     def deal_after_bidding(self):
#         host = self.host
#         cards = self.cards_shuffled[20:].reshape(4, 3)[:, 0:5]
#         cards = np.array([list2vec(l) for l in cards])
#         self.cards_vec[host] += cards[0]
#         for i in range(1, 4):
#             self.cards_vec[(host+i)%4] += cards[i]

#     def walkthrough(self):
#         self.deal_befor_bidding()
#         self.bidding()
#         self.deal_after_bidding()

#%%
## 打牌和发牌的模型分开训练，先训练打牌的模型，再训练发牌的模型
## Game 类用于训练打牌的模型，不考虑发牌，去掉发牌阶段
class Game:
    def __init__(self, cards=None, verbose=False):
        self.hokum = np.random.choice([0, 1, 2, 3, 4])  ## 0, 1, 2, 3 对应黑桃，红桃，梅花，方块，4 对应 sun
        self.host = np.random.choice(4)  ## 随机指定主叫方
        self.revealed_owner = self.host

        if cards is None:
            cards = np.array(range(32), dtype=np.int8) ## 洗牌
            np.random.shuffle(cards)
        cards = cards.reshape(4, 8)

        self.card_revealed = cards[self.host][0]  ## 明牌，发给主叫方
        self.cards_vec_init = np.array([list2vec(l) for l in cards])
        self.cards_vec = self.cards_vec_init.copy()
        self.history = []
        self.scores = []
        self.outof = np.zeros((4, 4), dtype=np.int8)
        self.verbose = verbose

    def register_bots(self, bots):
        self.bots = bots
        [bot.reset(self) for bot in self.bots]

    def oneround(self):
        history = {"host": self.host, "played": []}
        played = history["played"]
        self.history.append(history)
        for i in range(4):
            seat_current = (self.host + i) % 4
            choices = self.get_choices()
            card = self.bots[seat_current].play(self, choices)
            self.cards_vec[seat_current, card] = 0 ## 出牌
            played.append(card)

            ## 计算断牌
            color = card // 8
            if i == 0:
                color_the_round = color
            elif color != color_the_round:
                self.outof[seat_current, color_the_round] = 1
                if self.hokum < 4 and color != self.hokum:
                    self.outof[seat_current, self.hokum] = 1
            
            if self.verbose:
                print(f"seat {seat_current} played {cardstr[card]} ({list2str(choices)}) ({vec2str(self.cards_vec[seat_current])})")
        
        colors = [c // 8 for c in played]
        indices = [c % 8 for c in played]

        ## 计算本轮的赢家
        def get_newhost(colors, indices, hokum, host):
            color = colors[0]
            index = indices[0]
            newhost_offset = 0
            for i in range(1, 4):
                if colors[i] == color: ## 如果与当前最大牌花色相同
                    ranks = ranks_hokum if colors[i] == hokum else ranks_sun
                    if ranks[indices[i]] > ranks[index]: ## 如果比当前最大牌大
                        index = indices[i]
                        newhost_offset = i
                elif colors[i] == hokum: ## 如果此前最大牌不是hokum而这张牌是 hokum
                    color = colors[i]
                    index = indices[i]
                    newhost_offset = i
            return (host + newhost_offset) % 4
        self.host = get_newhost(colors, indices, self.hokum, self.host)

        ## 计算本轮分数
        def get_score_round(colors, indices, hokum):
            score = 0
            for color, index in zip(colors, indices):
                if color == hokum:
                    score += scores_hokum[index]
                else:
                    score += scores_sun[index]
            return score
        score = get_score_round(colors, indices, self.hokum)
        score = score if self.host % 2 == 0 else -score  ## 0,2 位次的玩家, 赢了加分, 输了减分. 更新后的 host 即为赢家
        self.scores.append(score)
        
        if self.verbose:
            print(f"Round:{len(self.history)} score:{score} Total Score:{np.array(self.scores).sum(axis=0)}")
        
    def get_choices(self):
        played = self.history[-1]["played"]
        seat = (self.host + len(played)) % 4
        cards = vec2list(self.cards_vec[seat])
        if len(played) == 0:  ## 先手，出任意牌
            choices = cards
        else:
            color = played[0] // 8
            choices = [card for card in cards if card // 8 == color] ## 出同花色牌
            if not choices:  ## 没有同花色牌
                if self.hokum < 4: ## hokum 局
                    choices = [card for card in cards if card // 8 == self.hokum] ## 出 hokum 牌
                    if not choices:  ## 没有 hokum 牌
                        choices = cards  ## 出任意牌
                else:  ## sun 局
                    choices = cards  ## 出任意牌
        return choices
        
    def whole_game(self):
        if self.verbose:
            if self.hokum < 4:
                print("hokum:", COLORS[self.hokum])
            else:
                print("hokum:", "sum")
        for i in range(8-len(self.history)):
            self.oneround()

#%%
class Game_Hokom(Game):
    def __init__(self, cards=None, verbose=False):
        super().__init__(cards=cards, verbose=verbose)
        self.hokum = 0

class Game_Sun(Game):
    def __init__(self, cards=None, verbose=False):
        super().__init__(cards=cards, verbose=verbose)
        self.hokum = 4


## Bot
## reset(): 重置Bot状态，接受一个参数 game. 训练模型时bot中需要保存一些数据，reset 用于重置这些数据
## play(): 出牌，接受一个参数 game
# %%
class Bot_Random:
    def __init__(self):
        pass

    def reset(self, game: Game):
        self.game = game

    def play(self, game: Game, choices):
        # choices = game.get_choices()
        return np.random.choice(choices)

# %%
class Bot_Rule:
    def __init__(self):
        pass

    def reset(self, game: Game):
        self.game = game

    ## 出牌
    ## 如果是本轮第一个出牌的，出最大的牌（按花色内的牌序）
    ## 如果是后手，如果有比场上更大的牌，或己方当前更大，出最大的牌，否则出最小的牌
    def play(self, game: Game, choices):
        if len(choices) == 1:
            return choices[0]
        h = game.history[-1]
        host, played = h["host"], h["played"]

        my_colors, my_ranks = [], []
        for card in choices:
            color = card // 8
            index = card % 8
            rank = ranks_hokum[index] if color == game.hokum else ranks_sun[index]
            my_colors.append(color)
            my_ranks.append(rank)

        if len(played) == 0:
            i = np.random.choice(np.where(np.array(my_ranks) == np.max(my_ranks))[0])
            return choices[i]

        color_round = played[0] // 8
        color_greatest = color_round
        index_greatest = played[0] % 8
        rank_greatest = ranks_hokum[index_greatest] if color_greatest == game.hokum else ranks_sun[index_greatest]
        order_greatest = 0

        for order, card in enumerate(played[1:]):
            order = order + 1
            color = card // 8
            index = card % 8
            rank = ranks_hokum[index] if color == game.hokum else ranks_sun[index]
            if color == color_greatest and rank > rank_greatest:
                rank_greatest = rank
                order_greatest = order
            elif color_greatest != game.hokum and color == game.hokum:
                color_greatest = game.hokum
                rank_greatest = rank
                order_greatest = order

        order = len(played)
        seat = (host + order) % 4
        hands = game.cards_vec[order]
        

        if my_colors[0] == color_greatest:  ## 我的花色和最大花色一样
            if np.max(my_ranks) > rank_greatest or order - order_greatest == 2:
                return choices[np.argmax(my_ranks)]
            else:
                return choices[np.argmin(my_ranks)]
        elif my_colors[0] == game.hokum:  ## 我的花色和最大花色不一样，和主花色一样
            return choices[np.argmax(my_ranks)]
        else:  ## 我的花色和最大花色不一样，和主花色也不一样
            if order - order_greatest == 2:
                i = np.random.choice(np.where(np.array(my_ranks) == np.max(my_ranks))[0])
                return choices[i]
            else:
                i = np.random.choice(np.where(np.array(my_ranks) == np.min(my_ranks))[0])
                return choices[i]

# # %%
# game = Game_Sun(verbose=True)
# # %%
# game.register([Bot_Random() for _ in range(4)])
# # %%
# game.whole_game()
# # %%
