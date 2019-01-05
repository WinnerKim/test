
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:25:53 2019

@author: hyokyeong
"""

# =============================================================================
# 이미지 RNN을 이용한 처리
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = False)
# 이미지 => RNN에 넣고 있음
# 28 * 28 => RNN적 특징을 추출하고 있음.

n_steps = 28 # 가로사이즈
n_inputs = 28 # 셀의 개수
n_neurons = 150 #특징을 몇개로 잡을 건지 결정
n_outputs = 10
n_layers = 3
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) # 150*28*28
y = tf.placeholder(tf.int32, [None])


# 셀이 수직으로 3개가 있음.
lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]

# 3층 셀을 수평으로 28개 만듬. (총 셀 28*3)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
ouputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)


top_layer_h_state = states[-1][1]
print(top_layer_h_state)


# FC(fully connected Neural network)로 연결해 주고 있음.
#dense : matrix를 알아서 계산해서 설정해줌.
#top_layer_h_state : 특징차수, n_outputs: 출력차수
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")


#예측값과 실제갑의 비교
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)


#손실값 구해줌. # mean값
loss = tf.reduce_mean(xentropy, name="loss")


# SGD(stochastic gb) : 확률적 경사하강법 / batch size(오차의 평균으로?) ====== 더 공부.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# 평가
correct = tf.nn.in_top_k(logits, y, 1) #index 반환해줌.
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# 실행
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})

        X_test=mnist.test.images.reshape(10000, n_steps, n_inputs)
        y_test=mnist.test.labels
        test_data = mnist.test.images[:batch_size].reshape((-1, n_steps, n_inputs))
        test_label = mnist.test.labels[:batch_size]
        acc_test = accuracy.eval(feed_dict={X: X_test, y:y_test})
        print("Epoch", epoch, "Train accuracy=", acc_train,
              "Test accuracy=", acc_test)

# =============================================================================




# =============================================================================
# word embedding
# https://3months.tistory.com/136(word embedding이란?)
import pprint
from tensorflow.python.framework import ops
import numpy as np
ops.reset_default_graph()
pp =pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
h = [1,0,0,0] # 4차원 벡터 encoding
e = [0,1,0,0] #
l = [0,0,1,0]
o = [0,0,0,1]

with tf.variable_scope('five_sequences') as scope:
    hidden_size=2
    # 4차원을 2차원 벡터로  embedding
    # R차원의 vector로 변환시켜 주는 것을 말함.
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    x_data = np.array([[h,e,l,l,o]], dtype=np.float32)
    print(x_data.shape) # 1,5,4
    pp.pprint(x_data)
    outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) # 5 by 2 로 나오고 있음.
# =============================================================================





# =============================================================================
# 끝단어 예측
ops.reset_default_graph()
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만듭니다.
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
# dictionary 만들고 있음.
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic) # 26

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
# wor -> X, d -> Y
# woo -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love',
            'kiss', 'kind']


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 여기서 생성하는 input_batch 와 target_batch 는
        # 알파벳 배열의 인덱스 번호 입니다.
        # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
        input = [num_dic[n] for n in seq[:-1]] # wor만 들어감. # 배열
        # 3, 3, 15, 4, 3 ...
        target = num_dic[seq[-1]]
        # one-hot 인코딩을 합니다.
        # if input is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])
        # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는
        # label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
        # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
        # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
        target_batch.append(target)

    return input_batch, target_batch

#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 30
# 타입 스텝: [1 2 3] => 3
# RNN 을 구성하는 시퀀스의 갯수입니다.
n_step = 3
# 입력값 크기. 알파벳에 대한 one-hot 인코딩이므로 26개가 됩니다.
# 예) c => [0 0 1 0 0 0 0 0 0 0 0 ... 0]
# 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류합니다.
n_input = n_class = dic_len

#########
# 신경망 모델 구성
######

X = tf.placeholder(tf.float32, [None, n_step, n_input]) # 10,3,26
# 비용함수에 sparse_softmax_cross_entropy_with_logits 을 사용하므로
# 출력값과의 계산을 위한 원본값의 형태는 one-hot vector가 아니라 인덱스 숫자를 그대로 사용하기 때문에
# 다음처럼 하나의 값만 있는 1차원 배열을 입력값으로 받습니다.
# [3] [3] [15] [4] ...
# 기존처럼 one-hot 인코딩을 사용한다면 입력값의 형태는 [None, n_class] 여야합니다.
Y = tf.placeholder(tf.int32, [None])


W = tf.Variable(tf.random_normal([n_hidden, n_class])) # 128*26
b = tf.Variable(tf.random_normal([n_class]))


# RNN 셀을 생성합니다.
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# 과적합 방지를 위한 Dropout 기법을 사용합니다.
# 일부 계산회로 제지
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
# 여러개의 셀을 조합해서 사용하기 위해 셀을 추가로 생성합니다.
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# 여러개의 셀을 조합한 RNN 셀을 생성합니다.
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

# tf.nn.dynamic_rnn 함수를 이용해 순환 신경망을 만듭니다.
# time_major=True
# outputs = 10,3,128
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# 최종 결과는 one-hot 인코딩 형식으로 만듭니다
outputs = tf.transpose(outputs, [1, 0, 2]) # 면과 행을 바꿈. = 3,10,128(RNN이 뽑은 특성)

# 연관된거라서 맨 마지막것만 보면 다 알 수 있음.
outputs = outputs[-1] # 1 * 10 * 128이 나옴.

# [1* 10* 128] * [128 *26]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())


input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch): # 30번 훈련함.
    _, loss = sess.run([optimizer, cost], # 그래프 꼭지를 돌리고?
                       feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

#########
# 결과 확인
######
# 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, Y: target_batch})


# 인덱스값으로 출력 = > 단어장에서 글자 확인.
predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)
# =============================================================================




# =============================================================================
# 번역 translate
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.

# 중복을 방지
ops.reset_default_graph()

# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현

import tensorflow as tf
import numpy as np

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]


# 번역망(왼쪽에 입력, 오른쪽에 번역, target = 번역과 target이 같아야함.)

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])



# 망이 두개임(인코더와 디코더) : initial_state를 통해서 연결.
# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states, # 인코더와 디코더 연결
                                            dtype=tf.float32)


model = tf.layers.dense(outputs, n_class, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated


print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))

########## NMT 모델 찾아볼 것.
# =============================================================================



# =============================================================================
# (tik tak toe) 게임
import csv
import random
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

response = 6
batch_size= 50
symmetry = ['rotate180','rotate90', 'rotate270', 'flip_v', 'flip_h']


def print_board(board):
    symbols = ['O', ' ', 'X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] +
          ' | ' + symbols[board_plus1[2]])
    print('___________')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] +
          ' | ' + symbols[board_plus1[5]])
    print('___________')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] +
          ' | ' + symbols[board_plus1[8]])


def get_symmetry(board, response, transformation):
    if transformation == 'rotate180':
        new_response = 8 - response
        return(board[::-1], new_response)
    elif transformation == 'rotate90':
        new_response = [6,3,0,7,4,1,8,5,2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return([value for item in tuple_board for value in item],
               new_response)
    elif transformation == 'rotate270':
        new_response = [2,5,8,1,4,7,0,3,6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return([value for item in tuple_board for value in item],
               new_response)
    elif transformation == 'flip_v': #012,345,678
        new_response = [6,7,8,3,4,5,0,1,2].index(response)
        return(board[6:9] + board[3:6] + board[0:3], new_response)
    elif transformation == 'flip_h':
        new_response = [2,1,0,5,4,3,8,7,6].index(response)
        new_board = board[::-1]
        return(new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)

    else:
        raise ValueError('해당하는 경우가 없음')




def get_moves_from_csv(csv_file):
    moves=[]
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter =',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return(moves)




def get_rand_move(moves, n=1, rand_transforms = 2):
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate180','rotate90', 'rotate270', 'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return(board, response)




moves = get_moves_from_csv('tictactoe_moves.csv')
train_length = 500
train_set = []
print(train_set)

for t in range(train_length):
    train_set.append(get_rand_move(moves))
print(len(train_set))
print(train_set)
test_board = [-1,0,0,1,-1,-1,0,0,1]
train_set = [x for x in train_set if x[0] != test_board]

def init_weights(shape):
    return(tf.Variable(tf.random_normal(shape)))

def model(X, A1, A2, bias1, bias2):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
    layer2 = tf.add(tf.matmul(layer1, A2), bias2)
    return(layer2)

X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
Y = tf.placeholder(dtype=tf.int32, shape=[None])
A1 = init_weights([9,81])
bias1 = init_weights([81])
A2 = init_weights([81,9])
bias2 = init_weights([9])
model_output = model(X, A1, A2, bias1, bias2)


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model_output, labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
prediction = tf.argmax(model_output, 1)
sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
for i in range(10000):
    rand_indices = np.random.choice(range(len(train_set)), batch_size, replace=False)
    batch_data = [train_set[i] for i in rand_indices]
    x_input = [x[0] for x in batch_data]
    y_target = np.array([y[1] for y in batch_data])
    sess.run(train_step, feed_dict={X: x_input, Y:y_target})
    temp_loss = sess.run(loss, feed_dict={X : x_input, Y: y_target})
    loss_vec.append(temp_loss)
    if i%500==0:
        print('iteration' + str(i) + 'Loss: ' + str(temp_loss))


import matplotlib.pyplot as plt
plt.plot(loss_vec, '-k', label='Loss')
plt.title('Loss (MSE) ')
plt.xlabel('Generaion')
plt.ylabel('Loss')
plt.show()


test_boards = [test_board]
feed_dict = {X : test_boards}
logits = sess.run(model_output, feed_dict = feed_dict)
predictions = sess.run(prediction, feed_dict=feed_dict)
print(predictions)



def check(board):
    wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
    for i in range(len(wins)):
        if board[wins[i][0]]==board[wins[i][1]]==board[wins[i][2]]==1.:
            return(1)
        elif board[wins[i][0]]==board[wins[i][1]]==board[wins[i][2]]==-1.:
            return(-1)
    return(0)


game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
win_logical = False
num_moves=0

while not win_logical:
    player_index = input('이동하고 싶은 인덱스를 입력하십시오 (0-8): ')
    num_moves += 1
    # Add player move to game
    game_tracker[int(player_index)] = 1.

    # Get model's move by first getting all the logits for each index
    [potential_moves] = sess.run(model_output, feed_dict={X: [game_tracker]})
    # Now find allowed moves (where game tracker values = 0.0)
    allowed_moves = [ix for ix, x in enumerate(game_tracker) if x == 0.0]
    # Find best move by taking argmax of logits if they are in allowed moves
    model_move = np.argmax([x if ix in allowed_moves else -999.0
                            for ix, x in enumerate(potential_moves)])

    # Add model move to game
    game_tracker[int(model_move)] = -1.
    print('모델이 이동하였습니다')
    print_board(game_tracker)
    # Now check for win or too many moves
    if check(game_tracker) == 1 or num_moves >= 20:
        print('게임종료! 승리하셨습니다')
        win_logical = True
    elif check(game_tracker)==-1:
        print('게임종료! 패배하셨습니다.')
        win_logical=True

# =============================================================================
