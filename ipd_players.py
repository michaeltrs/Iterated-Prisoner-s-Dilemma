# Players implememnting different iterated prisoner's dillemma strategies

import numpy as np
import matplotlib.pyplot as plt
from ipd_functions import round_choice, normalize
from scipy.stats import uniform, binom
from copy import deepcopy

# Score for each outcome:
# - rows   : p1 (0 cooperates, 1 defects)
# - columns: p2 (0 cooperates, 1 defects)
score_table = [[[-1, -1], [-3,  0]],
               [[ 0, -3], [-2, -2]]]


class Player:
    """
    General player superclass
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.score_history = []
        self.score = 0
    def update_score(self, p1, p2):
        out = score_table[p1][p2][self.player_id]
        self.score_history.append(out)
        self.score += out


class AlwaysDefect(Player):
    """
    A player that always chooses to defect (betray)
    the other player
    """
    name = 'always_defect'
    def __init__(self, player_id):
        Player.__init__(self, player_id)
    def choose(self, history):
        return 1


class AlwaysCooperate(Player):
    """
    A player that always chooses to cooperate with
    (show good faith to) the other player
    """
    name = 'always_cooperate'
    def __init__(self, player_id):
        Player.__init__(self, player_id)
    def choose(self, history):
        return 0


class Random_p(Player):
    """
    Stochastic algorithm
    A player that defects with probability prob_defect
    """
    name = 'random_p'
    def __init__(self, player_id, prob_defect):
        Player.__init__(self, player_id)
        self.prob_defect = prob_defect
    def choose(self, history):
        return int(np.random.rand() < self.prob_defect)


class Tic4Tak(Player):
    """
    Deterministic algorithm
    Cooperate at the first iteration then always do
    what th eother player did in the previous round
    Most robust basic deterministic strategy
    """
    name = 'tic_4_tak'
    def __init__(self, player_id):
        Player.__init__(self, player_id)
    def choose(self, history):
        if len(history) is 0:
            return 0
        else:
            return history[-1][1-self.player_id]

class Antony(Player):
    """
    Deterministic algorithm
    At first iteration defect
    At consequent iterations take the most probable choice
    from the opponent's history (mimic the opponent)
    """
    name = 'antony'
    def __init__(self, player_id):
        Player.__init__(self, player_id)
    def choose(self, history):
        if len(history) is 0:
            return 1
        else:
            history = np.array(history)
            return round_choice(history[:, 1-self.player_id].mean())


class Beth(Player):
    """
    Deterministic algorithm
    At first iteration defect
    At consequent iterations build a Bayesian model of the probability
    of the opponent defecting based on the opponent's history. If the mean
    probability is above opp_thres defect otherwise cooperate
        - uniform prior
        - binomial likelihood
    opp_thres : oppent threshold probability above which Beth defects
    """
    name = 'beth'
    def __init__(self, player_id, initial_choice, opp_thres):
        Player.__init__(self, player_id)
        self.opp_thres = opp_thres
        self.opponent_def_prob = np.linspace(0, 1, 1000)
        self.opponent_def_prob_prior = uniform(0, 1).pdf(self.opponent_def_prob)
        self.opponent_def_prob_post = uniform(0, 1).pdf(self.opponent_def_prob)
        self.opponent_def_prob_post_mean = 0.5
        self.opponent_def_prob_post_map = 0.5
        self.initial_choice = initial_choice
    def choose(self, history):
        # print("beth prior: ")
        # print(self.opponent_def_prob_prior)
        if len(history) is 0:
            return self.initial_choice
        else:
            history = np.array(history)[:,1-self.player_id]
            num_plays = len(history)
            num_def = len(history[history == 1])
            self.opponent_def_prob_post = normalize(
                binom(num_plays, self.opponent_def_prob).pmf(num_def) \
                * self.opponent_def_prob_prior)
            self.opponent_def_prob_post_mean = np.sum(self.opponent_def_prob_post * self.opponent_def_prob)
            #print("mean prob = ", prob_post_mean)
            self.opponent_def_prob_post_mean_map = self.opponent_def_prob[np.argmax(self.opponent_def_prob_post)]
            #print("map prob = ", prob_post_map)
            #print("opponent def prob: ", prob_post_mean)
            if self.opponent_def_prob_post_mean >= self.opp_thres:
                return 1
            else:
                return 0


class Beth2(Beth):
    """
    Same as Beth but the prior is taken as the posterior from the previous step
    """
    name = 'beth2'
    def __init__(self, player_id, initial_choice, opp_thres):
        Beth.__init__(self, player_id, initial_choice, opp_thres)
    def choose(self, history):
        # print("beth2 prior: ")
        # print(self.opponent_def_prob_prior)
        if len(history) is 0:
            return self.initial_choice
        else:
            history = np.array(history)[:,1-self.player_id]
            num_plays = len(history)
            num_def = len(history[history == 1])
            self.opponent_def_prob_post = normalize(
                binom(num_plays, self.opponent_def_prob).pmf(num_def) \
                * self.opponent_def_prob_prior)
            self.opponent_def_prob_post_mean = np.sum(self.opponent_def_prob_post * self.opponent_def_prob)
            self.opponent_def_prob_post_map = self.opponent_def_prob[np.argmax(self.opponent_def_prob_post)]
            # Update prior
            self.opponent_def_prob_prior = deepcopy(self.opponent_def_prob_post)
            if self.opponent_def_prob_post_mean >= self.opp_thres:
                #print(1)
                return 1
            else:
                #print(0)
                return 0


class Beth_stochastic_1(Beth):
    """
    Same as Beth2 but if the perceived opponent probbaility of defect is less than the specified threshold
    then there is a random probability of defect
    """
    name = 'beth_stochastic_1'
    def __init__(self, player_id, initial_choice, opp_thres):
        Beth.__init__(self, player_id, initial_choice, opp_thres)
    def choose(self, history):
        # print("beth2 prior: ")
        # print(self.opponent_def_prob_prior)
        if len(history) is 0:
            return self.initial_choice
        else:
            history = np.array(history)[:,1-self.player_id]
            num_plays = len(history)
            num_def = len(history[history == 1])
            self.opponent_def_prob_post = normalize(
                binom(num_plays, self.opponent_def_prob).pmf(num_def) \
                * self.opponent_def_prob_prior)
            self.opponent_def_prob_post_mean = np.sum(self.opponent_def_prob_post * self.opponent_def_prob)
            self.opponent_def_prob_post_map = self.opponent_def_prob[np.argmax(self.opponent_def_prob_post)]
            # Update prior
            self.opponent_def_prob_prior = deepcopy(self.opponent_def_prob_post)
            if self.opponent_def_prob_post_mean >= self.opp_thres:
                # guard yourself and protect
                return 1
            else:
                # randomly choose whether to
                return int(np.random.rand() < self.opponent_def_prob_post_mean)


class Beth_stochastic_2(Beth):
    """
    Same as Beth2 but if the perceived opponent probaility of defect is more than the specified threshold
    then there is a random probability of coooperation
    """
    name = 'beth_stochastic_2'
    def __init__(self, player_id, initial_choice, opp_thres):
        Beth.__init__(self, player_id, initial_choice, opp_thres)
    def choose(self, history):
        # print("beth2 prior: ")
        # print(self.opponent_def_prob_prior)
        if len(history) is 0:
            return self.initial_choice
        else:
            history = np.array(history)[:,1-self.player_id]
            num_plays = len(history)
            num_def = len(history[history == 1])
            self.opponent_def_prob_post = normalize(
                binom(num_plays, self.opponent_def_prob).pmf(num_def) \
                * self.opponent_def_prob_prior)
            self.opponent_def_prob_post_mean = np.sum(self.opponent_def_prob_post * self.opponent_def_prob)
            self.opponent_def_prob_post_map = self.opponent_def_prob[np.argmax(self.opponent_def_prob_post)]
            # Update prior
            self.opponent_def_prob_prior = deepcopy(self.opponent_def_prob_post)
            if self.opponent_def_prob_post_mean >= self.opp_thres:
                # guard yourself and protect
                return int(np.random.rand() < self.opponent_def_prob_post_mean)
            else:
                # randomly choose whether to
                return 0


if __name__ == "__main__":

    N = 100

    p1 = Beth_stochastic_1(0, 1, 0.5)
    #p1 = Antony(0)
    p2 = AlwaysDefect(1)
    #p2 = Random_p(1, 0.5)
    #p2 = AlwaysCooperate(1)

    history = []#np.zeros([N,2]).astype(int)

    for i in range(N):

        # Let the players choose
        p1_choice = p1.choose(history)
        p2_choice = p2.choose(history)

        # Update choices history
        history.append([p1_choice, p2_choice])
        #history[i] = [p1_choice, p2_choice]

        # Update players score
        p1.update_score(p1_choice, p2_choice)
        p2.update_score(p1_choice, p2_choice)


    print(p1.name, " score is: ", p1.score)
    print(p2.name, " score is: ", p2.score)

    plt.figure()
    plt.plot(np.cumsum(p1.score_history), label=p1.name)
    plt.plot(np.cumsum(p2.score_history), label=p2.name)
    plt.legend()
    plt.grid()

    plt.plot(p1.opponent_def_prob, p1.opponent_def_prob_post)

