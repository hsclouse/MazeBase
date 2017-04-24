from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
from time import sleep
from os import system
from cuda import *
from pprint import PrettyPrinter
from six.moves import input
import numpy as np
import argparse

import mazebase.games as games
from mazebase.games import featurizers
from mazebase.games import curriculum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import logging
logging.getLogger().setLevel(logging.DEBUG)

np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser(description='Policy Gradient RL in PyTorch')
parser.add_argument('--use_cuda', type=bool, default=False, metavar='C',
                    help='learning rate (default: 0.01)')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='R',
                    help='learning rate (default: 0.01)')
parser.add_argument('--std_dev', type=float, default=0.01, metavar='D',
                    help='standard deviation (default: 0.01)')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(args.seed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)


player_mode = False  # set this to True to use the uniform random choice action


class Policy(nn.Module):
    def __init__(self, data_size, hidden_size,output_size):
        super(Policy, self).__init__()
        self.linear = nn.Linear(data_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.linear.weight.data.normal_(0, args.std_dev)
        self.linear.bias.data.normal_(0, args.std_dev)
        self.linear2.weight.data.normal_(0, args.std_dev)
        self.linear2.bias.data.normal_(0, args.std_dev)
        self.data_size = data_size

        self.saved_actions = []
        self.rewards = []

    def forward(self, data):
        output = self.linear1(data)
        output = self.tanh(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output


switches = curriculum.CurriculumWrappedGame(
    games.Switches,
    waterpct=0.1, n_switches=4,
    switch_states=4,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (10, 10, 10, 10)
        )
    }
)
sg = curriculum.CurriculumWrappedGame(
    games.SingleGoal,
    waterpct=0.3,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (10, 10, 10, 10)
        )
    }
)
mg = curriculum.CurriculumWrappedGame(
    games.MultiGoals,
    waterpct=0.3,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (10, 10, 10, 10)
        )
    }
)
cg = curriculum.CurriculumWrappedGame(
    games.ConditionedGoals,
    waterpct=0.3,
    n_colors=4,
    n_goals=2,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (10, 10, 10, 10)
        )
    }
)
exclusion = curriculum.CurriculumWrappedGame(
    games.Exclusion,
    waterpct=0.2,
    n_goals=5,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (4, 4, 4, 4),
            (4, 4, 4, 4),
            (10, 10, 10, 10)
        ),
        'goal_penalty': games.curriculum.NumericCurriculum(
            0.4, 0.4, 1, lambda: randint(1, 5) / 20
        )
    }
)
pb = curriculum.CurriculumWrappedGame(
    games.PushBlock,
    waterpct=0.2,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (4, 4, 4, 4),
            (4, 4, 4, 4),
            (10, 10, 10, 10)
        )
    }
)
pbc = curriculum.CurriculumWrappedGame(
    games.PushBlockCardinal,
    waterpct=0.2,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (4, 4, 4, 4),
            (4, 4, 4, 4),
            (10, 10, 10, 10)
        )
    }
)
goto = curriculum.CurriculumWrappedGame(
    games.Goto,
    waterpct=0.2,
    blockpct=0.3,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (4, 4, 4, 4),
            (4, 4, 4, 4),
            (10, 10, 10, 10)
        )
    }
)
gotoh = curriculum.CurriculumWrappedGame(
    games.GotoHidden,
    waterpct=0.2,
    blockpct=0.1,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (10, 10, 10, 10)
        )
    }
)
lk = curriculum.CurriculumWrappedGame(
    games.LightKey,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (5, 5, 6, 6),
            (5, 5, 6, 6),
            (10, 10, 10, 10)
        ),
        'switch_states': games.curriculum.NumericCurriculum(2, 2, 5),
    }
)
bd = curriculum.CurriculumWrappedGame(
    games.BlockedDoor,
    curriculums={
        'map_size': games.curriculum.MapSizeCurriculum(
            (5, 5, 6, 6),
            (5, 5, 6, 6),
            (10, 10, 10, 10)
        ),
        'switch_states': games.curriculum.NumericCurriculum(2, 2, 5),
    }
)
all_games = [sg]
game = games.MazeGame(
    all_games,
    # featurizer=featurizers.SentenceFeaturesRelative(
    #   max_sentences=30, bounds=4)
    featurizer=featurizers.GridFeaturizer()
)
max_w, max_h = game.get_max_bounds()

pp = PrettyPrinter(indent=2, width=160)
# all_actions = game.all_possible_actions()
# all_features = game.all_possible_features()
# print("Actions:", all_actions)
# print("Features:", all_features)
# sleep(2)


def action_func(actions):
    if not player_mode:
        return choice(actions)
    else:
        print(list(enumerate(actions)))
        ind = -1
        while ind not in range(len(actions)):
            ind = input("Input number for action to take: ")
            try:
                ind = int(ind)
            except ValueError:
                ind = -1
        return actions[ind]


frame = 0
game.display()
sleep(.1)
system('clear')

actions = game.all_possible_actions()
config = game.observe()
obs, info = config['observation']
featurizers.grid_one_hot(game, obs)
obs = np.array(obs)

model = Policy(len(obs.flatten()), 128, len(actions))
if args.use_cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def select_action(actions, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    if args.use_cuda:
        probs = model(Variable(state.cuda()))
    else:
        probs = model(Variable(state))
    action = probs.multinomial()
    action = (action * 0) + np.argmax(probs.data.numpy())
    model.saved_actions.append(action)
    return actions[action.data.numpy()]

while True:
    print("r: {}\ttr: {} \tguess: {}".format(
        game.reward(), game.reward_so_far(), game.approx_best_reward()))
    config = game.observe()
    pp.pprint(config['observation'][1])
    # Uncomment this to featurize into one-hot vectors
    obs, info = config['observation']
    featurizers.grid_one_hot(game, obs)
    obs = np.array(obs)
    featurizers.vocabify(game, info)
    info = np.array(obs)
    config['observation'] = obs, info
    game.display()

    id = game.current_agent()
    actions = game.all_possible_actions()
    action = select_action(actions, obs)
    game.act(action)

    sleep(.1)
    system('clear')
    print("\n")
    frame += 1
    if game.is_over() or frame > 300:
        frame = 0
        print("Final reward is: {}, guess was {}".format(
            game.reward_so_far(), game.approx_best_reward()))
        game.make_harder()
        game.reset()
