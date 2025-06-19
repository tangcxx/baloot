## 训练价值网络

#%%
import os
import torch
from torch import nn
import numpy as np 
from datetime import datetime

import _game2
import _bot_v

# 训练参数
from _game2 import Game_Hokom, Bot_Random
from _bot_v import Model, Bot, Bot_Eval

# torch.set_num_threads(3)

modelpath = "model_v"
iterstart=0
model_freq = 1000
nmatch_per_iter = 16
nmatch_eval = 1000

gamma = 0.8
eps = 0.01
norm_var_init = torch.tensor(1200.0)
norm_alpha = 0.01
lossfn = nn.MSELoss()

def selfplay(models):
    bots = [Bot(models, eps=eps) for _ in range(4)]

    game = Game_Hokom()
    game.register_bots(bots)
    game.whole_game()

    ts = []
    t = 0
    for score in game.scores[::-1]:
        t = t * gamma + score
        ts.insert(0, t)
        
    zs, xs, Gs = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for seat, bot in enumerate(bots):
        for round, order, z, x in bot.data:
            zs[order].append(z)
            xs[order].append(x)
            
            Gs[order].append(ts[round] if seat % 2 == 0 else -ts[round])
    
    return zs, xs, Gs

def eval(models):
    scores = []
    for _ in range(nmatch_eval):
        cards = np.array(range(32), dtype=np.int8)
        np.random.shuffle(cards)

        bots = [Bot_Eval(models), Bot_Random(), Bot_Eval(models), Bot_Random()]
        game = Game_Hokom(cards=cards.copy())
        game.register_bots(bots)
        game.whole_game()
    
        bots = [Bot_Random(), Bot_Eval(models), Bot_Random(), Bot_Eval(models)]
        game2 = Game_Hokom(cards=cards.copy())
        game2.register_bots(bots)
        game2.whole_game()
        
        scores.append([np.sum(game.scores), -np.sum(game2.scores)])
    
    return np.array(scores).mean(axis=0), (np.array(scores) >= 0).mean(axis=0)

def checkpoint_save(iter, models, optizimers, norm_var):
    torch.save(
                    {
                        "models_state_dict": (models[0].state_dict(),
                                              models[1].state_dict(),
                                              models[2].state_dict(),
                                              models[3].state_dict()),
                        "optizimers_state_dict": (optizimers[0].state_dict(),
                                                  optizimers[1].state_dict(),
                                                  optizimers[2].state_dict(),
                                                  optizimers[3].state_dict()),
                        "norm_var": norm_var
                    },
                    f"{modelpath}/cp{iter}.pt"
                )

def checkpoint_load(iter, models, optimizers):
    cp = torch.load(f"{modelpath}/cp{iter}.pt")
    for i in range(4):
        models[i].load_state_dict(cp["models_state_dict"][i])
        optimizers[i].load_state_dict(cp["optizimers_state_dict"][i])
    return cp["norm_var"]

def train():
    iter = iterstart
    models = [Model(i) for i in range(4)]
    optimizers = [torch.optim.Adam(models[i].parameters(), lr=1e-4) for i in range(4)]
    norm_var = norm_var_init
    if iter == 0:
        os.makedirs(f"{modelpath}", exist_ok=True)
        checkpoint_save(iter, models, optimizers, norm_var)
    else:
        norm_var = checkpoint_load(iter, models, optimizers)
    
    f_log = open("{}/log.txt".format(modelpath), "a", buffering=1)
    f_eval = open("{}/eval.txt".format(modelpath), "a", buffering=1)
    while True:
        zs, xs, Gs = [[], [], [], []], [[], [], [], []], [[], [], [], []]
        for _ in range(nmatch_per_iter):
            zs_match, xs_match, Gs_match = selfplay(models)
            for i in range(4):
                zs[i].extend(zs_match[i])
                xs[i].extend(xs_match[i])
                Gs[i].extend(Gs_match[i])

        norm_var = norm_var * (1 - norm_alpha) + torch.tensor(Gs[0]).var() * norm_alpha

        for zs_i, xs_i, Gs_i, model, optimizer in zip(zs, xs, Gs, models, optimizers):
            gs_i = torch.tensor(Gs_i, dtype=torch.float32)
            gs_i = gs_i/ norm_var.sqrt()  # Normalize returns
            gs_i = gs_i.reshape(-1, 1)
            zs_i = torch.cat(zs_i)
            xs_i = torch.cat(xs_i)
            
            loss = lossfn(model(zs_i, xs_i), gs_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, models, optimizers, norm_var)
            scores, wins = eval(models)
            f_eval.write(f"{iter}\t{nmatch_eval}\t{scores[0]}\t{scores[1]}\t{wins[0]}\t{wins[1]}\n")
            print(datetime.now(), "eval:", scores[0], scores[1], wins[0], wins[1])
            f_log.write(f"{datetime.now()}\t{iter}\n")
            print(datetime.now(), iter)



#%%
if __name__ == '__main__':
   train()
