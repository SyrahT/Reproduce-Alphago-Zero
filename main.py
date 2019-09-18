
from go import *
import tensorflow as tf
import random
import sys
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
        input_array = trans_recent_to_array(self.board.recent, False)
        p = pred_p_v(input_array)
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

def pred_p_v(XX):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        #saver = tf.train.import_meta_graph('my_test_model.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./'))
        p = sess.run(pred_p, feed_dict={X:XX, is_training:False})
        return  p

def trans_recent_to_array(recent, is_training):

    if len(recent) == 0:
        return np.zeros([1, board_size, board_size, 9])
    
    now_board = Position(board=None, n=0, komi=1, caps=(0, 0), lib_tracker=None, ko=None, recent=tuple(), to_play=BLACK)
    add_one_game = []
    last = -8 if is_training == 1 else len(recent)-9
    for i in range(last,len(recent)-8):
        now_board = Position(board=None, n=0, komi=1, caps=(0, 0), lib_tracker=None, ko=None, recent=tuple(), to_play=BLACK) if i<0 else now_board.play_move(c=recent[i].move, color=recent[i].color)
        sub_board = now_board
        add = now_board.board
        for j in range(1,8):
            if i+j>=0:
                sub_board = sub_board.play_move(c=recent[i+j].move, color=recent[i+j].color)
            add = np.dstack((add, sub_board.board))

        add = np.dstack((add, np.ones(now_board.board.shape)*recent[i+8].color))
        if add_one_game == []:
            add_one_game = add[np.newaxis,:]
        else:
            add_one_game = np.vstack((add_one_game, add[np.newaxis,:]))
    return add_one_game

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

def expand(node):

    tried_sub_node_moves = [sub_node.state.board.recent[-1].move for sub_node in node.children]
    probability = node.state.get_policy_through_net()
    probability = np.reshape(probability, [board_size * board_size])
    probability = probability / np.max(probability)
    probability = np.power(probability, 1/tempreture)
    probability = probability * 0.75 + 0.25 * np.random.randn(board_size * board_size)

    for i in range(board_size):
        for j in range(board_size):
            if ((i,j)  not in node.state.all_choices) or probability[i * board_size + j]<0:
                probability[i*board_size + j] = 0
    probability = probability / np.sum(probability)
    move_ = np.random.choice(board_size * board_size, 1, p=probability)
    move = (move_[0] //  board_size, move_[0] %  board_size)

    if move in tried_sub_node_moves:
        for sub_node in node.children:
            if sub_node.state.board.recent[-1].move == move:
                return sub_node
    else:
        board_copy = copy.deepcopy(node.state.board)
        new_position = board_copy.play_move(move)
        new_state = State(new_position)
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

def monte_carlo_tree_search(node):
    #print('monte_carlo_tree_search')
    computation_budget = 100
    for i in range(computation_budget):
        expand_node = tree_policy(node)
        reward = default_policy(expand_node)
        backup(expand_node, reward)

    best_next_node = best_child(node, False)
    if best_next_node == None:
        return node
    return best_next_node

def self_play():
    current_node = initial_node
    for i in range(1000):
        print("Play round: {}".format(i + 1))
        current_node = monte_carlo_tree_search(current_node)
        if len(current_node.state.all_choices) == 0 or current_node.state.board.n <= i:
            break
    return current_node.state.board.recent, current_node.state.board.win_or_lose()

def trans_recent_to_policy(recent):

    now_node = initial_node
    add_one_game_policy = []
    for i in range(len(recent)):
        add = np.zeros((board_size, board_size))
        bo = 1
        for child in now_node.children:
            if child.state.board.recent[-1].move == recent[i].move and bo:
                next_node = child
                bo = 0
            add[child.state.board.recent[-1].move[0]][child.state.board.recent[-1].move[1]] =  child.visit_times

        add = add / np.sum(add)
        if add_one_game_policy == []:
            add_one_game_policy = add[np.newaxis,:]
        else:
            add_one_game_policy = np.vstack((add_one_game_policy, add[np.newaxis,:]))
        now_node = next_node
    add_one_game_policy= np.reshape(add_one_game_policy,[-1,25])
    return add_one_game_policy

def rotate(a, b, c):
    aa = np.zeros(a.shape)
    cc = np.zeros(c.shape)
    for times in range(4):
        for i in range(board_size):
            for j in range(board_size):
                for l in range(a.shape[0]):
                    for k in range(9):
                        aa[l][j][board_size-1-i][k] = a[l][i][j][k]
                    cc[l][j*board_size + board_size-1-i] = c[l][i*board_size+j]
        net_update(aa, b, cc)

def reflect(a, b, c):
    for l in range(len(b)):
        for i in range(board_size):
            for j in range(int(board_size/2)):
                tmp = a[l][i][j]
                a[l][i][j] = a[l][i][board_size-1-j]
                a[l][i][board_size-1-j] = tmp
                tmp = c[l][i*board_size+j]
                c[l][i*board_size+j] = c[l][i*board_size+board_size-1-j]
                c[l][i*board_size+board_size-1-j] = tmp
    return a, b, c

def residual_block(f0, _weights, _biases, is_training, number, X_residual):

    a1_ = tf.nn.conv2d(f0, _weights['Wconv'+str(number)], strides=[1,1,1,1], padding='SAME') + _biases['bconv'+str(number)]
    a1_ = tf.cast(a1_, dtype= 'float32')
    b1_ = tf.layers.batch_normalization(a1_, training=is_training)
    f1_ = tf.nn.relu(b1_)
    f1_ = tf.cast(f1_, dtype= 'float64')

    add = tf.concat([f1_, X_residual], 3)
    a1 = tf.nn.conv2d(add, _weights['Wconv'+str(number)+'_'], strides=[1,1,1,1], padding='SAME') + _biases['bconv'+str(number)+'_']
    a1 = tf.cast(a1, dtype= 'float32')
    b1 = tf.layers.batch_normalization(a1, training=is_training)
    b1 = tf.cast(b1, dtype= 'float64')
    f1 = tf.nn.relu(b1)
    return f1

def network(X, _weights, _biases, is_training):

    X_residual = X
    a0 = tf.nn.conv2d(X, _weights['Wconv'], strides=[1,1,1,1], padding='SAME') + _biases['bconv']
    a0 = tf.cast(a0, dtype= 'float32')
    b0 = tf.layers.batch_normalization(a0, training=is_training)
    f0 = tf.nn.relu(b0)
    f0 = tf.cast(f0, dtype= 'float64')

    for i in range(1):
        f1 = residual_block(f0,  _weights, _biases, is_training, i, X_residual)

    ap = tf.nn.conv2d(f1, _weights['Wp'], strides=[1,1,1,1], padding='SAME') + _biases['bp']
    ap = tf.cast(ap, dtype= 'float32')
    bp = tf.layers.batch_normalization(ap, training=is_training)
    bp = tf.cast(bp, dtype= 'float64')
    cp = tf.nn.relu(bp)
    cp = tf.reshape(cp, shape=[-1, board_size * board_size * 256])         
    fp = tf.layers.dense(inputs=cp, units= board_size * board_size, activation=tf.nn.sigmoid)
    fp = tf.nn.sigmoid(fp)
    #print(fp)

    av = tf.nn.conv2d(f1, _weights['Wv'], strides=[1,1,1,1], padding='SAME') + _biases['bv']
    av = tf.cast(av, dtype= 'float32')
    bv = tf.layers.batch_normalization(av, training=is_training)
    bv = tf.cast(bv, dtype= 'float64')
    cv = tf.nn.relu(bv)
    cv = tf.reshape(cv, shape=[-1, board_size * board_size * 256])  
    fv = tf.layers.dense(inputs=cv, units= board_size * board_size, activation=tf.nn.relu)
    fv = tf.layers.dense(inputs=fv, units= 1, activation=tf.nn.tanh)
    #print(fv)

    return fp, fv

def net_update(X_train, z_train, policy_train):
    length = len(z_train)
    z_train = np.array(z_train)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        #saver = tf.train.import_meta_graph('my_test_model.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./'))

        step = 1
        while step * batch_size < training_iters:
            mask = np.random.choice(length, size=batch_size, replace=False)
            X_batch = X_train[mask,:,:,:]
            y_batch = z_train[mask]
            y_p_batch = policy_train[mask,:]
            sess.run(optimizer, feed_dict={X: X_batch, y_v: y_batch, y_p:y_p_batch, is_training:True})

            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: X_batch, y_v: y_batch, is_training:False})
                los = sess.run(loss, feed_dict={X: X_batch, y_v: y_batch, y_p:y_p_batch, is_training:False})
                print ("Iter " + str(step*batch_size) \
                    + ", Minibatch Loss= " + "{:.6f}".format(los) \
                    + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        saver.save(sess, 'my_test_model')
        print ("Optimization Finished!")

def play(node, complete_or_fast):
    #---------computer moves--------

    if complete_or_fast:

        node = monte_carlo_tree_search(node)

    else:

        tried_sub_node_moves = [sub_node.state.board.recent[-1].move for sub_node in node.children]
        probability = node.state.get_policy_through_net()
        probability = np.reshape(probability, [board_size * board_size])
        probability = probability / np.max(probability)
        probability = np.power(probability, 1/tempreture)
        probability = probability * 0.75 + 0.25 * np.random.randn(board_size * board_size)

        for i in range(board_size):
            for j in range(board_size):
                if ((i,j)  not in node.state.all_choices) or probability[i * board_size + j]<0:
                    probability[i*board_size + j] = 0
        probability = probability / np.sum(probability)
        move_ = np.random.choice(board_size * board_size, 1, p=probability)
        move = (move_[0] //  board_size, move_[0] %  board_size)

        if move in tried_sub_node_moves:
            for sub_node in node.children:
                if sub_node.state.board.recent[-1].move == move:
                    node = sub_node
        else:
            board_copy = copy.deepcopy(node.state.board)
            new_position = board_copy.play_move(move)
            new_state = State(new_position)
            sub_node = Node(state = new_state)
            node.add_child(sub_node)
            node = sub_node

    print(node.state.board)
    #---------people moves-----------
    print('Your Turn:')
    c1 = input('Move1:')
    c2 = input('Move2:')
    flag = True
    move = (c1, c2)
    for sub_node in node.children:
        if sub_node.state.board.recent[-1].move == move:
            node = sub_node
            flag = False
            break
    if flag:
        new_board = node.state.board
        new_board.play_move(move)
        new_state = State(new_board)
        sub_node = Node(state = new_state)
        node.add_child(sub_node)
        node = sub_node
    print(node.state.board)

    #----------------------------------

    play(node, complete_or_fast)

def play_initial(complete_or_fast):
    black_or_white = input('Black or white:')

    my_board = Position(board=None, n=0, komi=1, caps=(0, 0), lib_tracker=None, ko=None, recent=tuple(), to_play=BLACK)
    root_state = State(my_board)
    root = Node(state = root_state)

    if black_or_white:
        c1 = input('Move1:')
        c2 = input('Move2:')
        my_board.play_move((c1, c2))

    play(root, complete_or_fast)


if __name__ == "__main__":

    board_size = 9
    set_board_size(board_size)
    tempreture = 1
    #------------tensorflow network--------------

    learning_rate = 0.0001
    training_iters = 2000
    batch_size = 10
    display_step = 5

    tf.reset_default_graph()
    weights = {
        'Wconv' : tf.Variable(tf.random_normal([3,3,9,256],dtype=np.float64)),
        'Wconv0' : tf.Variable(tf.random_normal([3,3,256,256],dtype=np.float64)),
        'Wconv0_' : tf.Variable(tf.random_normal([3,3,265,256],dtype=np.float64)),
        'Wp' : tf.Variable(tf.random_normal([1,1,256,256],dtype=np.float64)),
        'Wv' : tf.Variable(tf.random_normal([1,1,256,256],dtype=np.float64)),
        'Wconv1' : tf.Variable(tf.random_normal([3,3,256,256],dtype=np.float64)),
        'Wconv1_' : tf.Variable(tf.random_normal([3,3,265,256],dtype=np.float64)),
        'Wconv2' : tf.Variable(tf.random_normal([3,3,256,256],dtype=np.float64)),
        'Wconv2_' : tf.Variable(tf.random_normal([3,3,265,256],dtype=np.float64)),
        'Wconv3' : tf.Variable(tf.random_normal([3,3,256,256],dtype=np.float64)),
        'Wconv3_' : tf.Variable(tf.random_normal([3,3,265,256],dtype=np.float64)),
        'Wconv4' : tf.Variable(tf.random_normal([3,3,256,256],dtype=np.float64)),
        'Wconv4_' : tf.Variable(tf.random_normal([3,3,265,256],dtype=np.float64))
        
    }
    biases = {
        'bconv' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv0' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv0_' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bp' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bv' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv1' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv1_' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv2' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv2_' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv3' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv3_' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv4' : tf.Variable(tf.random_normal([256],dtype=np.float64)),
        'bconv4_' : tf.Variable(tf.random_normal([256],dtype=np.float64))
    }

    X = tf.placeholder(tf.float64, [None, board_size, board_size, 9])
    y_v = tf.placeholder(tf.float64, [None])
    y_p = tf.placeholder(tf.float64, [None, board_size * board_size])
    is_training = tf.placeholder(tf.bool)

    pred_p, pred_v= network(X, weights, biases, is_training)
    loss = -tf.reduce_mean(tf.multiply(y_p, tf.log(pred_p))) + tf.reduce_mean(tf.square(y_v - pred_v))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    ones = tf.ones_like(pred_v)
    negative_ones = - ones
    correct_pred = tf.equal(tf.where(pred_v>0.5, ones, negative_ones), y_v)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float64))

    #-------------self play and reinforcement--------------
    my_board = Position(board=None, n=0, komi=1, caps=(0, 0), lib_tracker=None, ko=None, recent=tuple(), to_play=BLACK)
    initial_state = State(my_board)
    initial_node = Node(state = initial_state)

    for times in range(1):
        recent, w_or_l = self_play()
        if w_or_l != 0:
            add_one_game = trans_recent_to_array(recent, True)
            z = []
            for i in range(-8,len(recent)-8):
                z.append(w_or_l * recent[i+8].color)
            add_one_game_policy = trans_recent_to_policy(recent)

            rotate(add_one_game, z, add_one_game_policy)
            add_one_game, z, add_one_game_policy = reflect(add_one_game, z, add_one_game_policy)
            rotate(add_one_game, z, add_one_game_policy)

    print('Well Done!')
    #--------------play against computer-----------------
    '''print('Start Play:')
    complete_or_fast = input('complete(0)_or_fast(1):')
    play(initial_node, complete_or_fast)'''


