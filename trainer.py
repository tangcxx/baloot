# 基于fairtrainer_aug
# bot模型修改:
# lstm 输出取h_n的最后一个时刻的输出
# lstm 输入取最后30手出牌

#%%
import os
import torch
from torch import nn
import numpy as np 
import multiprocessing as mp
from datetime import datetime

import _game
import _bot

# 训练参数
from _game import Game, Bot_Random
from _bot import Bot, Model

modelpath = "model"
iterstart=0
model_freq = 5
nmatch_per_iter = 128
nmatch_eval = 100

model_subs = [Model(0), Model(1), Model(2), Model(3)]

def selfplay(models):
    bots = [Bot(models=models) for _ in range(4)]

    game = Game()
    game.register_bots(bots)
    game.whole_game()

    scores = np.add.accumulate(game.scores[::-1])[-2::-1]
    log_probs = [[] for _ in range(4)]
    Gs = [[] for _ in range(4)]
    for bot in bots:
        for i in range(7):
            order, log_prob, seat = bot.data[i]
            score = scores[i] if seat % 2 == 0 else -scores[i]
            log_probs[order].append(log_prob)
            Gs[order].append(score)

    return log_probs, Gs

def eval(models):
    score0 = 0
    for _ in range(nmatch_eval):
        bots = [Bot(models=models), Bot_Random(), Bot(models=models), Bot_Random()]
        game = Game()
        game.register_bots(bots)
        game.whole_game()
        score0 = score0 + np.sum(game.scores)
    
    score1 = 0
    for _ in range(nmatch_eval):
        bots = [Bot_Random(), Bot(models=models), Bot_Random(), Bot(models=models)]
        game = Game()
        game.register_bots(bots)
        game.whole_game()
        score1 = score1 - np.sum(game.scores)
    
    return score0/nmatch_eval, score1/nmatch_eval

def checkpoint_save(iter, models, optizimers):
    torch.save(
                    {
                        "models_state_dict": (models[0].state_dict(), models[1].state_dict(), models[2].state_dict(), models[3].state_dict()),
                        "optizimers_state_dict": (optizimers[0].state_dict(), optizimers[1].state_dict(), optizimers[2].state_dict(), optizimers[3].state_dict())
                    },
                    f"{modelpath}/cp{iter}.pt"
                )

def checkpoint_load(iter, models, optimizers):
    cp = torch.load(f"{modelpath}/cp{iter}.pt")
    for order in range(4):
        models[order].load_state_dict(cp["models_state_dict"][order])
        optimizers[order].load_state_dict(cp["optizimers_state_dict"][order])

def train():
    iter = iterstart
    models = [Model(0), Model(1), Model(2), Model(3)]
    optimizers = [torch.optim.Adam(models[order].parameters(), lr=1e-4) for order in range(4)]
    if iter == 0:
        os.makedirs(f"{modelpath}", exist_ok=True)
        checkpoint_save(iter, models, optimizers)
    else:
        checkpoint_load(iter, models, optimizers)
    
    f_log = open("{}/log.txt".format(modelpath), "a", buffering=1)
    f_eval = open("{}/eval.txt".format(modelpath), "a", buffering=1)
    while True:
        res = [selfplay(models) for _ in range(nmatch_per_iter)]

        log_probs_orders, Gs_orders = [[], [], [], []], [[], [], [], []]
        for log_probs, Gs in res:
            for order in range(4):
                log_probs_orders[order].extend(log_probs[order])
                Gs_orders[order].extend(Gs[order])

        for log_probs, Gs, optimizer in zip(log_probs_orders, Gs_orders, optimizers):
            loss = []
            for log_prob, score in zip(log_probs, Gs):
                loss.append(-log_prob * score/76)
            optimizer.zero_grad()
            loss = torch.stack(loss).sum()
            loss.backward()
            optimizer.step()

        # assert False

        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, models, optimizers)
            score0, score1 = eval(models)
            f_eval.write(f"{iter}\t{nmatch_eval}\t{score0}\t{score1}\n")
            print(datetime.now(), "eval:", score0, score1)
        print(datetime.now(), iter)
        f_log.write(f"{datetime.now()}\t{iter}\n")



#%%
if __name__ == '__main__':
   train()
