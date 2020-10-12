# encoding=utf-8
'''
Author: Haitaifantuan
Create Date: 2020-09-08 23:47:11
Author Email: 47970915@qq.com
Description: Should you have any question, do not hesitate to contact me via E-mail.
'''
import numpy as np
import random
import time


class SARSA(object):
    def __init__(self):
        self.total_rows = 4
        self.total_columns = 4
        self.total_action_num = 4  # 0代表上，1代表右，2代表下，3代表左
        self.reward_each_step = -1
        self.action_dict = {0: '上', 1: '右', 2: '下', 3: '左'}
        self.reversed_action_dict = {'上': 0, '右':1, '下':2, '左': 3}
        # 分别是走上、下、左、右的概率。随机数命中某个数字如49，那就是向右。随机数只在0-100随机选数字。
        self.four_action_probability = {'上': range(0, 25), '右': range(25, 50), '下': range(50, 75), '左': range(75, 100)}
        self.idx_change_dict = {'上': (-1, 0), '右': (0, 1), '下': (1, 0), '左': (0, -1)}  # 左边这个是行的索引的改变，右边这个是列的索引的改变
        self.episode = 10000000  # 共采集TOTAL_ITERATION幕数据

        self.initial_epsilon = 0.5
        self.epsilon_decay_ratio = 0.90
        self.final_epsilon = 0.01

        self.alpha = 0.8
        self.gamma = 0.9


        # 初始化q函数，也就是状态动作价值函数
        # 一共16个状态，每个状态4个动作，因此一共64个值。并且终止状态的动作价值都为0
        # 我们将其都设置为0，形状是（4，4，4）
        self.q_function = np.zeros((self.total_rows, self.total_columns, 4))
        print("q_function为：{}".format(self.q_function))
        print("============================================================")

        # 初始化状态价值函数V
        maze = np.zeros((self.total_rows, self.total_columns))  # 用0代表迷宫的每一格，这个maze只是方便我们看迷宫，没有其他作用。
        print("maze为：{}".format(maze))
        print("============================================================")


    def go_one_step_and_get_current_reward_and_next_state(self, current_state, action):
        '''
        根据当前的状态，以及行为，计算当前行为的奖励以及下一个状态
        '''
        row_idx, column_idx = current_state

        # 计算下下一步的state和reward
        next_row_idx = row_idx + self.idx_change_dict[action][0]
        next_column_idx = column_idx + self.idx_change_dict[action][1]

        # 先判断是否到了终点，如果是终点，不管执行什么操作
        # 奖励都是0，并且都会回到终点

        if (next_row_idx == 0 and next_column_idx == 0):
            return 0, (0, 0)

        if (next_row_idx == 3 and next_column_idx == 3):
            return 0, (3, 3)

        # 再判断是否在边缘，如果是的话，那就回到该位置。
        if next_row_idx < 0 or next_row_idx > self.total_rows - 1 or next_column_idx < 0 or next_column_idx > self.total_columns - 1:
            return self.reward_each_step, (row_idx, column_idx)
        else:
            return self.reward_each_step, (next_row_idx, next_column_idx)


    def generate_initial_state(self, total_rows, total_columns):
        row_idx = random.randint(0, total_rows - 1)
        column_idx = random.randint(0, total_columns - 1)

        while (row_idx == 0 and column_idx == 0) or (row_idx == 3 and column_idx == 3):
            row_idx = random.randint(0, total_rows - 1)
            column_idx = random.randint(0, total_columns - 1)

        return (row_idx, column_idx)


    def fire_calculation(self):
        # 对每一个episode进行循环
        self.epsilon = self.initial_epsilon

        for episode in range(self.episode):
            # 随机生成一个起始状态
            init_state = self.generate_initial_state(self.total_rows, self.total_columns)
            current_state = init_state
            current_state_row_idx, current_state_column_idx = init_state

            # 随机选取一个动作
            if random.random() < self.epsilon:
                current_state_max_action_idx = random.choice(list(self.action_dict.keys()))
            else:
                # 在这里，下一个动作应该根据策略去选择的，也就是根据概率去选择，这里我们简化了下，我们直接选取值函数最大的值对应的动作
                # 根据q函数决定做什么动作，实际上q函数在这个例子里是一个表格。如果有多个相同的最大值，那就在里面随机选取
                max_value = np.max(self.q_function[current_state_row_idx][current_state_column_idx])
                current_state_max_action_idx = random.choice(
                    np.where(self.q_function[current_state_row_idx][current_state_column_idx] == max_value)[0])

            # 根据q函数以及当前的状态做一个动作，我们这里采用ε-greedy的方法。
            # 遍历每一个步
            while not ((current_state_row_idx == 0 and current_state_column_idx == 0) or (current_state_row_idx == 3 and current_state_column_idx == 3)):
                # 执行该动作
                action = self.action_dict[current_state_max_action_idx]
                # 根据要走的动作得到奖励以及获取下一个状态
                current_state_action_reward, next_state = self.go_one_step_and_get_current_reward_and_next_state(current_state, action)
                next_state_row_idx, next_state_column_idx = next_state

                # 继续根据q函数以及next_state，决定出next_state情况下，该执行什么动作，这里也采用ε-greedy的方法
                if random.random() < self.epsilon:
                    next_state_max_action_idx = random.choice(list(self.action_dict.keys()))
                else:
                    # 在这里，下一个动作应该根据策略去选择的，也就是根据概率去选择，这里我们简化了下，我们直接选取值函数最大的值对应的动作
                    max_value = np.max(self.q_function[next_state_row_idx][next_state_column_idx])
                    next_state_max_action_idx = random.choice(
                        np.where(self.q_function[next_state_row_idx][next_state_column_idx] == max_value)[0])

                # 根据公式更新q函数
                self.q_function[current_state_row_idx][current_state_column_idx][current_state_max_action_idx] = \
                    self.q_function[current_state_row_idx][current_state_column_idx][current_state_max_action_idx] + \
                    self.alpha * (current_state_action_reward + self.gamma * \
                                  self.q_function[next_state_row_idx][next_state_column_idx][next_state_max_action_idx] - \
                                  self.q_function[current_state_row_idx][current_state_column_idx][current_state_max_action_idx])

                # 将下一个状态赋值给当前状态
                current_state_row_idx, current_state_column_idx = next_state_row_idx, next_state_column_idx
                current_state = (current_state_row_idx, current_state_column_idx)
                # 将提前计算出来的下一个动作赋值给当前动作
                current_state_max_action_idx = next_state_max_action_idx

            # 一个episode结束后更新epsilon的值
            if self.epsilon > self.final_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay_ratio

            if episode % 10000 == 0:
                print("第{}个episode已经结束".format(episode))


    def show_policy(self):
        self.policy = {}

        shape = np.shape(self.q_function)
        for row_idx in range(shape[0]):
            for column_idx in range(shape[1]):
                for action_idx in range(shape[2]):
                    state = str(row_idx) + str(column_idx)
                    if state not in self.policy.keys():
                        self.policy[state] = {self.action_dict[action_idx]: self.q_function[row_idx][column_idx][action_idx]}
                    else:
                        self.policy[state][self.action_dict[action_idx]] = self.q_function[row_idx][column_idx][action_idx]

        print(self.policy)


obj = SARSA()
obj.fire_calculation()
obj.show_policy()