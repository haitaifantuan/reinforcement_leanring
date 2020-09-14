# encoding=utf-8
'''
Author: Haitaifantuan
Create Date: 2020-09-07 23:10:17
Author Email: 47970915@qq.com
Description: Should you have any question, do not hesitate to contact me via E-mail.
'''
import numpy as np
import copy


TOTAL_ROWS = 4
TOTAL_COLUMN = 4
TOTAL_ACTIONS_NUM = 4  # 0代表上，1代表右，2代表下，3代表左
STOP_CRITERION = 1e-4
GAMMA = 1
REWARD_SETTING = -1
ACTION_DICT = {0: '上', 1: '右', 2: '下', 3: '左'}
FOUR_ACTION_PROBABILITY = {'上': 0.25, '右': 0.25, '下': 0.25, '左': 0.25}  # 分别是走上、下、左、右的概率。
IDX_CHANGE_DICT = {'上': (-1, 0), '右': (0, 1), '下': (1, 0), '左': (0, -1)}  # 左边这个是行的索引的改变，右边这个是列的索引的改变

def get_current_reward_and_next_state(current_state, action):
    '''
    根据当前的状态，以及行为，计算当前行为的奖励以及下一个状态
    '''
    # 先判断是否到了终点，如果是终点，不管执行什么操作
    # 奖励都是0，并且都会回到终点
    row_idx, column_idx = current_state
    if (row_idx == 0 and column_idx == 0):
        return 0, (0, 0)

    if (row_idx == 3 and column_idx == 3):
        return 0, (3, 3)

    # 否则的话就计算下下一步的state和reward
    next_row_idx = row_idx + IDX_CHANGE_DICT[action][0]
    next_column_idx = column_idx + IDX_CHANGE_DICT[action][1]

    # 再判断是否在边缘，如果是的话，那就回到该位置。
    if next_row_idx < 0 or next_row_idx > TOTAL_ROWS - 1 or next_column_idx < 0 or next_column_idx > TOTAL_COLUMN - 1:
        return REWARD_SETTING, (row_idx, column_idx)
    else:
        return REWARD_SETTING, (next_row_idx, next_column_idx)


# 初始化状态价值函数V
V = np.zeros((TOTAL_ROWS, TOTAL_COLUMN))

# 开始迭代更新状态价值函数
iteration = 0
flag = True
while flag:
    delta = 0
    old_V = copy.deepcopy(V)
    # 遍历每一个状态，对其进行更新
    for row_idx in range(TOTAL_ROWS):
        for column_idx in range(TOTAL_COLUMN):
            new_final_value = 0
            # 根据sutton的《强化学习》第72页公式4.5进行更新
            for each_action in range(TOTAL_ACTIONS_NUM):
                action = ACTION_DICT[each_action]
                action_proba = FOUR_ACTION_PROBABILITY[action]
                current_action_reward, next_state = get_current_reward_and_next_state((row_idx, column_idx), action)
                new_final_value = new_final_value + action_proba * (1 * (current_action_reward +
                                                                         GAMMA * V[next_state[0]][next_state[1]]))

            V[row_idx][column_idx] = new_final_value


    delta = max(delta, abs(old_V - V).max())

    if delta < STOP_CRITERION:
        flag = False

    iteration += 1

print(V)
print(iteration)