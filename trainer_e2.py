# 一组牌 embedding 之后, 增加一层 MLP , 然后再求和

#%%
import os
import torch
from torch import nn
from datetime import datetime

import _game_torch
import _bot_e2

# 训练参数
from _game_torch import Game_Hokom, Bot_Random
from _bot_e2 import Model, Bot, Bot_Eval

gamma = 0.8

modelpath = "model_e2"
iterstart=0
model_freq = 1000
nmatch_per_iter = 20
nmatch_eval = 1000

torch.set_num_threads(1)

def selfplay(models):
    bots = [Bot(models) for _ in range(4)]

    game = Game_Hokom()
    game.register_bots(bots)
    game.whole_game()

    ts = []
    t = 0
    for score in game.scores[::-1]:
        t = t * gamma + score
        ts.insert(0, t)
        
    log_probs, Gs, vs = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for bot in bots:
        for round, order, log_prob, seat, v in bot.data:
            Gs[order].append(ts[round] if seat % 2 == 0 else -ts[round])
            log_probs[order].append(log_prob)
            vs[order].append(v)

    return log_probs, Gs, vs

def eval(models):
    scores = torch.zeros((nmatch_eval, 2), dtype=torch.float32)
    for i in range(nmatch_eval):
        cards_init = torch.randperm(32).reshape(4, 8).tolist()
        cards_init2 = [cards[:] for cards in cards_init]  ## 深拷贝，避免修改原始数据

        bots = [Bot_Eval(models), Bot_Random(), Bot_Eval(models), Bot_Random()]
        game = Game_Hokom(cards_init=cards_init)
        game.register_bots(bots)
        game.whole_game()
    
        bots = [Bot_Random(), Bot_Eval(models), Bot_Random(), Bot_Eval(models)]
        game2 = Game_Hokom(cards_init=cards_init2)
        game2.register_bots(bots)
        game2.whole_game()
        
        scores[i] = torch.tensor([sum(game.scores), -sum(game2.scores)])

    return torch.mean(scores, dim=0), torch.mean((scores>0).float(), dim=0)

def checkpoint_save(iter, models, optizimers):
    torch.save(
                    {
                        "models_state_dict": (models[0].state_dict(),
                                              models[1].state_dict(),
                                              models[2].state_dict(),
                                              models[3].state_dict()),
                        "optizimers_state_dict": (optizimers[0].state_dict(),
                                                  optizimers[1].state_dict(),
                                                  optizimers[2].state_dict(),
                                                  optizimers[3].state_dict())
                    },
                    f"{modelpath}/cp{iter}.pt"
                )

def checkpoint_load(iter, models, optimizers):
    cp = torch.load(f"{modelpath}/cp{iter}.pt")
    for i in range(4):
        models[i].load_state_dict(cp["models_state_dict"][i])
        optimizers[i].load_state_dict(cp["optizimers_state_dict"][i])

def train():
    iter = iterstart
    models = [Model(i) for i in range(4)]
    optimizers = [torch.optim.Adam(models[i].parameters(), lr=1e-4) for i in range(4)]
    if iter == 0:
        os.makedirs(f"{modelpath}", exist_ok=True)
        checkpoint_save(iter, models, optimizers)
    else:
        checkpoint_load(iter, models, optimizers)
    
    f_log = open("{}/log.txt".format(modelpath), "a", buffering=1)
    f_eval = open("{}/eval.txt".format(modelpath), "a", buffering=1)
    while True:
        log_probs, Gs, vs = [[], [], [], []], [[], [], [], []], [[], [], [], []]
        for _ in range(nmatch_per_iter):
            log_probs_match, Gs_match, vs_match = selfplay(models)
            for i in range(4):
                log_probs[i].extend(log_probs_match[i])
                Gs[i].extend(Gs_match[i])
                vs[i].extend(vs_match[i])

        for log_probs_i, Gs_i, vs_i, optimizer in zip(log_probs, Gs, vs, optimizers):
            gs_i = torch.tensor(Gs_i, dtype=torch.float32)
            gs_i = (gs_i - gs_i.mean()) / (gs_i.std() + 1e-8)  # Normalize returns
            log_probs_i = torch.stack(log_probs_i).squeeze()
            vs_i = torch.stack(vs_i).squeeze()
            
            policy_loss = -log_probs_i * (gs_i - vs_i).detach() 
            value_loss = nn.functional.mse_loss(vs_i, gs_i)  # Policy gradient loss + value loss
            loss = policy_loss.sum() + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, models, optimizers)
            scores, wins = eval(models)
            scores, wins = scores.tolist(), wins.tolist()
            f_eval.write(f"{iter}\t{nmatch_eval}\t{scores[0]}\t{scores[1]}\t{wins[0]}\t{wins[1]}\n")
            print(datetime.now(), "eval:", scores[0], scores[1], wins[0], wins[1])
            f_log.write(f"{datetime.now()}\t{iter}\n")
            print(datetime.now(), iter)



#%%
if __name__ == '__main__':
   train()
