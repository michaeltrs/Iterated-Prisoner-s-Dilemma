# Environment for testing iterated prisoner's dillema Players

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipd_players import *


def define_player(p, player_id):
    if p in [Beth, Beth2, Beth_stochastic_1, Beth_stochastic_2]:
        initial_choice = players_settings[p][0]
        opp_thres = players_settings[p][1]
        p = p(player_id, initial_choice, opp_thres)
    else:
        p = p(player_id)
    return p


def one_vs_one(p1, p2, N, plot_res=False):
    history = []  # np.zeros([N,2]).astype(int)

    for iter_ in range(N):
        # Let the players choose
        p1_choice = p1.choose(history)
        p2_choice = p2.choose(history)

        # Update choices history
        history.append([p1_choice, p2_choice])
        # history[i] = [p1_choice, p2_choice]

        # Update players score
        p1.update_score(p1_choice, p2_choice)
        p2.update_score(p1_choice, p2_choice)

    if plot_res:
        plt.figure()
        plt.plot(np.cumsum(p1.score_history), label=p1.name)
        plt.plot(np.cumsum(p2.score_history), label=p2.name)
        plt.legend()
        plt.grid()

    return p1, p2


class All_vs_all:

    def __init__(self, players, players_settings, N, plot_res=False):
        self.players = players
        self.players_names = [player.name for player in players]
        self.players_settings = players_settings
        self.num_players = len(self.players)
        self.N = N
        self.plot_res = plot_res

        self.score_table = self.run()

    def run(self):
        # list of dicts
        # games_results = []
        # dataframe
        score_table = pd.DataFrame(np.zeros((self.num_players, self.num_players)),
                                     columns=self.players_names,
                                     index=self.players_names)
        for i in range(self.num_players):
            for j in range(self.num_players):
                if i==j:
                    pass
                else:
                    p1 = define_player(self.players[i], 0)
                    p2 = define_player(self.players[j], 1)

                    p1, p2 = one_vs_one(p1, p2, N, self.plot_res)
                    score_table[p1.name][p2.name] = p1.score
                    # if j > i:
                    #     games_results.append({p1.name:p1.score, p2.name:p2.score})
        return score_table

    def get_mean_scores(self):
        return self.score_table.sum(axis=0)/(self.num_players-1)


if __name__ == "__main__":

    players = [AlwaysDefect, AlwaysCooperate, Tic4Tak, Antony, Beth2, Beth_stochastic_2]
    players_settings = {Beth: [0, 0.5], Beth2: [0, 0.5], Beth_stochastic_1:[0, 0.5], Beth_stochastic_2:[0, 0.5]}
    N = 1000

    game = All_vs_all(players, players_settings, N)

    score_table = game.score_table

    mean_scores = game.get_mean_scores()

    print("And the Winner is: ", np.argmax(mean_scores))


