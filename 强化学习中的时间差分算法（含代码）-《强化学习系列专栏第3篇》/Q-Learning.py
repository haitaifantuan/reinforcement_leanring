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


class Q_Learning(object):
    def __init__(self):
        # 创建一个q函数，其实就是Q表格
        # 一共6个状态，每个状态都有合法的action
        # 这个列代表当前所处的状态，行代表即将到达的状态
        self.q_function = np.zeros((6, 6))
        # 这个字典记录了在哪个状态（房间），可以到达哪个状态（房间）
        self.available_action = {0:[4], 1:[3, 5], 2:[3], 3:[1, 2, 4], 4:[0, 3], 5:[1]}
        self.destination_reward = 100
        self.episode = 10000  # 共采集TOTAL_ITERATION幕数据
        self.initial_epsilon = 0.5
        self.epsilon_decay_ratio = 0.90
        self.final_epsilon = 0.01
        self.alpha = 0.8
        self.gamma = 0.9
        print("q_function为：{}".format(self.q_function))
        print("============================================================")


    def generate_initial_state(self):
        state = random.randint(0, 5)
        while state == 0 or state == 5:
            state = random.randint(0, 5)
        return state


    def generate_action_and_get_reward_and_next_state(self, current_state):
        # 使用ε-greedy的方法以及贪婪的方法选取一个动作以及做一个动作并获得回报
        if random.random() < self.epsilon:
            # 根据当前的状态，随机选取一个动作
            next_state = random.choice(self.available_action[current_state])
        else:
            # 找到当前可到达的下一个状态
            available_next_state_list = self.available_action[current_state]
            # 根据可到达的下一个状态，使用贪婪的策略找到使得获得最大收益的那个动作对应的下一个状态
            # 拿到可行的下一个状态的回报
            available_next_state_reward_list = self.q_function[current_state][available_next_state_list]
            # 找到最大的那个回报的索引，如果有多个同样大小的值，就随机选取一个
            max_value = np.max(available_next_state_reward_list)
            indices = list(np.where(available_next_state_reward_list == max_value)[0])
            idx = random.choice(indices)
            # 根据这个索引找到最大回报的下一个状态是什么
            next_state = available_next_state_list[idx]

        # 判断下是否到达4号房间或者5号房间
        if next_state == 4 or next_state == 5:
            reward = 100
            finished_flag = True
        else:
            reward = 0
            finished_flag = False


        return next_state, reward, finished_flag


    def fire_calculation(self):
        # 对每一个episode进行循环
        self.epsilon = self.initial_epsilon

        for episode in range(self.episode):
            # 随机生成一个起始状态
            current_state = self.generate_initial_state()
            finished_flag = False

            while not finished_flag:
                # 使用ε-greedy的方法以及贪婪的方法选取一个动作以及做一个动作并获得回报
                next_state, current_state_reward, finished_flag = self.generate_action_and_get_reward_and_next_state(current_state)

                # 根据下一个状态，使用ε-greedy的方法以及贪婪的方法选取一个动作以及做一个动作并获得回报
                # 这里不需要管finished_flag是否为True，因为我们只是根据next_state拿到next_state情况下最大的下下个状态的奖励值
                next_next_state, next_state_reward, next_state_finished_flag = self.generate_action_and_get_reward_and_next_state(next_state)

                # 根据公式更新q函数
                self.q_function[current_state][next_state] = self.q_function[current_state][next_state] + \
                    self.alpha * (current_state_reward + self.gamma * (next_state_reward) - self.q_function[current_state][next_state])

                current_state = next_state

            # 一个episode结束后更新epsilon的值
            if self.epsilon > self.final_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay_ratio

            if episode % 3000 == 0:
                print("第{}个episode已经结束".format(episode))
                print("当前的q_function是：{}".format(self.q_function))


    def show_policy(self):
        print("当前的q_function是：{}".format(self.q_function))
        self.policy = {}

        print("策略是：")
        shape = np.shape(self.q_function)
        for current_state in range(shape[0]):
            max_reward_state = np.argmax(self.q_function[current_state])
            self.policy[current_state] = max_reward_state
            if current_state == 4 or current_state == 5:
                continue
            print("如果你在{}号房间，那就往{}号房间走".format(current_state, max_reward_state))



obj = Q_Learning()
obj.fire_calculation()
obj.show_policy()