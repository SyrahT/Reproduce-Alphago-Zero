
from go import *
import random
import sys
import tensorflow as tf
import copy
import math

class State():
    def __init__(self, position = None):
        self.board = position
        self.all_choices = None if position == None else position.get_choices()

    def is_terminal(self):
        return self.board.is_terminal()

    def get_next_state_randomly(self):
        random_choice = random.choice(self.all_choices)
        board_copy = copy.deepcopy(self.board)
        new_position = board_copy.play_move(random_choice)
        next_state = State(new_position)
        return next_state, random_choice

    def get_policy_through_net(self):
        input_array = trans_recent_to_array(self.board.recent, False, self.board.board.shape[0])
        #print(input_array)
        p, v = pred_p_v(input_array)
        return p


class Node():
    def __init__(self, parent = None, children = [], visit_times = 0, quality_value = 0.0, state = None):
        self.parent = parent
        self.children = children
        self.visit_times = visit_times
        self.quality_value = quality_value
        self.state = state

    def set_parent(self, parent):
        self.parent = parent

    def is_all_expand(self):
        return len(self.children) >= len(self.state.all_choices)

    def add_child(self, child):
        self.children = [child] + self.children
        #a big problem // using self.children.append(child) will create a loop
        child.set_parent(self)


def tree_policy(node):
    while node.state.is_terminal() == False:
        if node.is_all_expand():
            node = best_child(node, True)
        else:
            sub_node = expand(node)
            return sub_node
    return node

def default_policy(node):
    current_state = node.state
    while current_state.is_terminal() == False:
        current_state, nothing_ = current_state.get_next_state_randomly()
    final_state_reward = current_state.board.win_or_lose() * current_state.board.to_play
    return final_state_reward

def get_next_move(probability):
    p = np.random.uniform(0,1)
    all_p = 0
    for i in range(board_size):
        for j in range(board_size):
            all_p += probability[i][j]
            if all_p>p:
                return (i,j)

def get_rank(array):
    rank = []
    for i in range(board_size):
        for j in range(board_size):
            rank.append((i,j))
    for i in range(len(rank)):
        for j in range(len(rank)):
            if array[rank[i][0],rank[i][1]]<array[rank[j][0],rank[j][1]]:
                tmp = rank[i]
                rank[i] = rank[j]
                rank[j] = tmp
    return rank

def get_move(array):
    array = array / np.sum(array)
    print(array)
    p = np.random.random()
    all_p = 0.0
    for i in range(board_size):
        for j in range(board_size):
            all_p += array[i][j]
            if all_p>p:
                return (i,j)

def expand(node):

    tried_sub_node_moves = [sub_node.state.board.recent[-1].move for sub_node in node.children]
    probability = node.state.get_policy_through_net()
    probability = probability / np.max(probability)
    probability = np.power(probability, 1/tempreture)
    probability = probability * 0.75 + 0.25 * np.random.random()

    for i in range(board_size):
        for j in range(board_size):
            if (i,j) in node.state.all_choices == False:
                probability[i][j] = 0
                    
    move = get_move(probability)
    if move in tried_sub_node_moves:
        for sub_node in node.children:
            if sub_node.state.board.recent[-1].move == move:
                node = sub_node
                break

    sub_node = Node(state = new_state)
    node.add_child(sub_node)
    return sub_node


def best_child(node, is_exploration):#UCB
    to_play = node.state.board.to_play
    best_score = -sys.maxsize * to_play
    best_sub_node = None
    for sub_node in node.children:
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0

        left = sub_node.quality_value / sub_node.visit_times
        right = 2.0 * math.log(node.visit_times) / sub_node.visit_times
        score = left + C * math.sqrt(right)

        if score * to_play > best_score * to_play:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node

def backup(node, reward):
    while node != None:
        node.visit_times += 1
        node.quality_value += reward * node.state.board.to_play
        node = node.parent

def check(node):
    list_node = []
    list_node.append(node)
    while len(list_node)>0:
        check_now = list_node.pop(0)
        print(len(check_now.children))
        for sub_node in check_now.children:
            list_node.append(sub_node)
            if check_now.state.board.n>sub_node.state.board.n:
                print("Contradiction!")

def monte_carlo_tree_search(node):
    computation_budget = 100
    for i in range(computation_budget):
        expand_node = tree_policy(node)
        reward = default_policy(expand_node)
        backup(expand_node, reward)

    best_next_node = best_child(node, False)
    return best_next_node


