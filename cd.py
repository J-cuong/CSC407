# -*- coding: utf-8 -*-
'''
Reference code for data loading.
Author: Chongyang Bai
For more details, refer to the paper:
C.Bai, S. Kumar, J. Leskovec, M. Metzger, J.F. Nunamaker, V.S. Subrahmanian,
Predicting Visual Focus of Attention in Multi-person Discussion Videos,
International Joint Conference on Artificial Intelligence (IJCAI), 2019.
'''

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadGame(game, N):
    # N is the number of players in the game
    # load csv data
    # unweighted and weighted networks can be loaded exactly the same way
    # below shows the loader for weighted networks
    df_network = pd.read_csv(f'{src}/network{game}_weighted.csv', index_col=0)

    # T is number of timestamps (10 frames)
    T = len(df_network)
    # load VFOA network to T x N x (N+1) array
    # vfoa[t, n, i] is the probability of player n+1 looking at object i at time t
    # i: 0 - laptop, 1 - player 1, 2 - player 2, ..., N - player N
    vfoa = np.reshape(df_network.values, (T, N, N + 1))

    # print information
    print(f'network id:{game}\t length(x 1/3 second): {T}\t num of players: {N}')
    return vfoa

###
# Assumptions:
# 1. Each step allows each person to look at only one thing
# 2. face-to-face interaction means looking at someone
# 3. p = P(transmission | looking)
# 4. Each look at each step could transmit a rumour


'''
Returns how many players heard rumour at end of game

params:
    game: looking probabilities
    p: probability of hearing rumour
    N: number of players
'''


def simluate_game(game, N, p):

    # Heard list marks which player heard rumour
    # i: 0 - laptop, 1 - player 1, 2 - player 2, ..., N - player N
    heard = [0] * (N + 1)

    # Initalize one player with a rumour
    heard[random.randint(1, N)] = 1

    # Iterate through each step in the last third of the game
    for step in range(int(len(game) * 2 / 3), len(game)):

        # Iterate through each potential rumour sender
        for sender in range(1, N + 1):

            # If the sender heard the rumour...
            if heard[sender] == 1:

                # ... find the probability of the sender looking at the reciever
                look_probs = list(game[step][sender - 1])

                # Genereate the index of the recieve based on looking probabilities
                receiver = int(np.random.choice(list(range(N + 1)), 1, look_probs))

                # With probability of p, have the reciever recieve the rumour
                for prob in look_probs:
                    if random.random() < p*prob:
                        heard[receiver] = 1

        return sum(heard[1::])


'''
Simluates game over different p's and finds people who heard rumours
'''


def get_heard_versus_p(game, N):
    X, Y = [], []
    for i in range(10):
        p = 10**(-i)
        tot_heard = []
        for _ in range(10):
            tot_heard.append(simluate_game(game, N, p))
        X.append(p)
        Y.append(np.average(tot_heard))
    return X, Y


for x in range(1,11): 
    print("Weighted trial: " + str(x))
    src = './network'  # root dir of data
    N = 8  # number of players
    game = loadGame(5, N)
    X, Y = get_heard_versus_p(game, N)
    print(X)
    print(Y)
    print("\n" + "\n")

print("\n" + "\n")

colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(X, Y, s=area, c=colors, alpha=0.9)
plt.title('Graph @ 2/3')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, Y, '.r-')
plt.show()

def loadGame_unweighted(game, N):
    # N is the number of players in the game
    # load csv data
    df_network = pd.read_csv(f'{src}/network{game}.csv', index_col=0)

    # T is number of timestamps (10 frames)
    T = len(df_network)
    # load VFOA network to T x N x (N+1) array
    # vfoa[t, n, i] is the probability of player n+1 looking at object i at time t
    # i: 0 - laptop, 1 - player 1, 2 - player 2, ..., N - player N
    vfoa = np.reshape(df_network.values, (T, N, N + 1))

    # print information
    print(f'network id:{game}\t length(x 1/3 second): {T}\t num of players: {N}')
    return vfoa


for x in range(1,11): 
    print("Un-weighted trial: " + str(x))
    src = './network'  # root dir of data
    N = 8  # number of players
    game = loadGame_unweighted(5, N)
    X, Y = get_heard_versus_p(game, N)
    print(X)
    print(Y)
    print("\n" + "\n")

colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(X, Y, s=area, c=colors, alpha=0.9)
plt.title('Graph @ 2/3')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, Y, '.r-')
plt.show()
