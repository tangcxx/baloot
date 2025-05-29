# 基于fairtrainer_aug
# bot模型修改:
# lstm 输出取h_n的最后一个时刻的输出
# lstm 输入取最后30手出牌

#%%
import os
import torch
from torch import nn
import numpy as np 
from datetime import datetime

import _game
import _bot_b

# 训练参数
from _game import Game, Bot_Random
from _bot_b import Bot, Model

gamma = 1

modelpath = "model_b"
iterstart=73000
model_freq = 1000
nmatch_per_iter = 1
nmatch_eval = 1000

def selfplay(model):
    bots = [Bot(model) for _ in range(4)]

    game = Game()
    game.register_bots(bots)
    game.whole_game()

    ts = []
    t = 0
    for score in game.scores[::-1]:
        t = t * gamma + score
        ts.insert(0, t)
        
    log_probs = []
    Gs = []
    for bot in bots:
        for round, order, log_prob, seat in bot.data:
            Gs.append(ts[round] if seat % 2 == 0 else -ts[round])
            log_probs.append(log_prob)

    return log_probs, Gs

def eval(model):
    score0 = 0
    for _ in range(nmatch_eval):
        bots = [Bot(model), Bot_Random(), Bot(model), Bot_Random()]
        game = Game()
        game.register_bots(bots)
        game.whole_game()
        score0 = score0 + np.sum(game.scores)
    
    score1 = 0
    for _ in range(nmatch_eval):
        bots = [Bot_Random(), Bot(model), Bot_Random(), Bot(model)]
        game = Game()
        game.register_bots(bots)
        game.whole_game()
        score1 = score1 - np.sum(game.scores)
    
    return score0/nmatch_eval, score1/nmatch_eval

def checkpoint_save(iter, model, optizimer):
    torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optizimer_state_dict": optizimer.state_dict()
                    },
                    f"{modelpath}/cp{iter}.pt"
                )

def checkpoint_load(iter, model, optimizer):
    cp = torch.load(f"{modelpath}/cp{iter}.pt")
    model.load_state_dict(cp["model_state_dict"])
    optimizer.load_state_dict(cp["optizimer_state_dict"])

def train():
    iter = iterstart
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if iter == 0:
        os.makedirs(f"{modelpath}", exist_ok=True)
        checkpoint_save(iter, model, optimizer)
    else:
        checkpoint_load(iter, model, optimizer)
    
    f_log = open("{}/log.txt".format(modelpath), "a", buffering=1)
    f_eval = open("{}/eval.txt".format(modelpath), "a", buffering=1)
    while True:
        log_probs, Gs = [], []
        for _ in range(nmatch_per_iter):
            u, v = selfplay(model)
            log_probs.extend(u)
            Gs.extend(v)

        Gs = torch.tensor(Gs, dtype=torch.float32)
        Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-9)
        loss = []
        for log_prob, g in zip(log_probs, Gs):
            loss.append(-log_prob * g)
        optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        optimizer.step()


        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, model, optimizer)
            score0, score1 = eval(model)
            f_eval.write(f"{iter}\t{nmatch_eval}\t{score0}\t{score1}\n")
            print(datetime.now(), "eval:", score0, score1)
            print(datetime.now(), iter)
            f_log.write(f"{datetime.now()}\t{iter}\n")



#%%
if __name__ == '__main__':
   train()
