from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
from time import sleep
import os
from os import system
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
logging.basicConfig(filename="debug.log")
logging.getLogger().setLevel(logging.DEBUG)

np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser(description='Policy Gradient RL in PyTorch')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--use_cuda', action='store_true',
                    help='enable CUDA')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='R',
                    help='learning rate (default: 0.01)')
parser.add_argument('--std_dev', type=float, default=0.01, metavar='D',
                    help='standard deviation (default: 0.01)')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--player', action='store_false',
                    help='enables user input to game')
parser.add_argument('--store_rewards', type=str, default=" ",
                    help='writes game scores to file')
args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(args.seed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)

player_mode = args.player


class Policy(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(data_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear15 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.linear1.weight.data.normal_(0, args.std_dev)
        self.linear1.bias.data.normal_(0, args.std_dev)
        self.linear15.weight.data.normal_(0, args.std_dev)
        self.linear15.bias.data.normal_(0, args.std_dev)
        self.linear2.weight.data.normal_(0, args.std_dev)
        self.linear2.bias.data.normal_(0, args.std_dev)
        self.data_size = data_size

        self.saved_actions = []
        self.rewards = []

    def forward(self, data):
        output = F.relu(self.linear1(data))
        # output = self.tanh(output)
        output = F.relu(self.linear15(output))
        output = F.relu(self.linear2(output))
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
    waterpct=0,
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

if args.render:
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
if args.render:
    game.display()
    sleep(.1)
    system('clear')

actions = game.all_possible_actions()
config = game.observe()
obs, info = config['observation']
featurizers.grid_one_hot(game, obs)
obs = np.array(obs)

model = Policy(len(obs.flatten()), 128, len(actions))
# model.load_state_dict(torch.load('./model_chkpt_00047.pth'))  # load weights
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
    model.saved_actions.append(action)
    # action.cpu()
    return actions[action.data.numpy()[0][0]]


def finish_episode(episode):
    R = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(model.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    # Uncomment below to store model checkpoints
    # if episode % args.log_interval == 0:  # save the model every so often
    #     chkpt_num = int(episode)
    #     torch.save(model.state_dict(), './model_chkpt_{:07d}.pth'.format(chkpt_num))
    #     print("Checkpoint {} saved.".format(chkpt_num))

episode_num = 0
while True:
    if args.render:
        print("r: {}\ttr: {} \tguess: {}".format(
            game.reward(), game.reward_so_far(), game.approx_best_reward()))
    config = game.observe()
    if args.render:
        pp.pprint(config['observation'][1])
    # Uncomment this to featurize into one-hot vectors
    obs, info = config['observation']
    featurizers.grid_one_hot(game, obs)
    obs = np.array(obs)
    featurizers.vocabify(game, info)
    info = np.array(obs)
    config['observation'] = obs, info
    if args.render:
        game.display()

    id = game.current_agent()
    actions = game.all_possible_actions()
    action = select_action(actions, obs.flatten())
    if args.render:
        print("Action: " + action + "\n")
    game.act(action)
    model.rewards.append(game.reward())

    if args.render:
        # sleep(.1)
        system('clear')
        print("\n")
    frame += 1
    if game.is_over() or frame > 300:
        frame = 0
        episode_num += 1
        # if episode_num > 20:  # ran out of space during training so only keep last 20 chkpts
        #     episode_num = 0
        reward_so_far = game.reward_so_far()
        best_reward = game.approx_best_reward()
        if args.render:
            print("Final reward is: {}, guess was {}".format(
                reward_so_far, best_reward))
        finish_episode(episode_num)
        if args.store_rewards is not " ":
            with open(args.store_rewards, "a+") as f:
                f.write(str(reward_so_far) + "\t" + str(best_reward) + "\n")
        # if episode_num % 400) == 0:
        #     game.make_harder()
        #     print("Difficulty Increased.\n")
        game.reset()
