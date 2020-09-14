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


class First_Visit_Monte_Carlo_Policy_Evaluation(object):
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
        self.episode = 100000  # 共采集TOTAL_ITERATION幕数据

        # 初始化状态价值函数V
        maze = np.zeros((self.total_rows, self.total_columns))  # 用0代表迷宫的每一格，这个maze只是方便我们看迷宫，没有其他作用。
        print(maze)


    def get_current_reward_and_next_state(self, current_state, action):
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


    def generate_one_episode_data(self, init_state):
        one_episode_data = []
        current_state = init_state
        while not ((current_state[0] == 0 and current_state[1] == 0) or (current_state[0] == 3 and current_state[1] == 3)):
            # 根据概率产生一个动作
            rand_int = random.randint(0, 99)
            for each in self.four_action_probability.items():
                if rand_int in each[1]:
                    action = each[0]
                    break

            # 根据要走的动作得到奖励以及获取下一个状态
            reward, next_state = self.get_current_reward_and_next_state(current_state, action)
            # （当前状态，当前行为，当前行为的奖励）
            one_episode_data.append((current_state, self.reversed_action_dict[action], reward))

            current_state = next_state

        # while循环出来的时候，最后一个terminal状态没加进去。
        one_episode_data.append((current_state, None, None))

        return one_episode_data


    def fire_calculation(self):
        # 计算“状态-动作价值”
        # 创建一个字典保存出现的状态-动作以及奖励
        begin_time = time.time()

        episode_record_dict = {}
        final_state_action_reward_dict = {}  # 这个和state_action_reward_dict是有区别的，它是记录总体的。
        final_state_action_count_dict = {}
        for episode in range(self.episode):
            # 生成每一幕的数据
            all_episode_list = []
            # 随机生成一个起始状态
            init_state = self.generate_initial_state(self.total_rows, self.total_columns)
            # 生成一幕数据
            episode_record_dict[episode] = self.generate_one_episode_data(init_state)
            # 对这幕数据进行遍历，然后将出现过的状态进行统计
            has_been_counted = {}  # 记录状态是否在该幕出现过
            state_action_reward_dict = {}  # 记录每一个状态当前的总共的reward
            for idx, eachTuple in enumerate(episode_record_dict[episode]):
                # 先判断是不是到了终点，如果是的话就跳出循环
                if idx == len(episode_record_dict[episode])-1:
                    break

                # 将state和action组合成字符串，方便作为dict的key
                state_action_combination = str(eachTuple[0][0]) + str(eachTuple[0][1]) + str(eachTuple[1])

                # 对state_action_reward_dict()里的所有的key都累加当前的reward。
                for key in state_action_reward_dict.keys():
                    state_action_reward_dict[key] += eachTuple[2]

                # 检测当前这一幕该状态和动作组合是否出现过
                if state_action_combination not in has_been_counted.keys():
                    # 如果不存在在state_count_dict.keys()里，说明是第一次碰到该状态。
                    has_been_counted[state_action_combination] = 1  # 随便赋值一个value
                    state_action_reward_dict[state_action_combination] = eachTuple[2]

            # 将该募最后统计到总的变量里。
            for state_action, reward in state_action_reward_dict.items():
                if state_action not in final_state_action_reward_dict.keys():
                    final_state_action_reward_dict[state_action] = reward  # 将该状态-动作计数设为reward
                    final_state_action_count_dict[state_action] = 1  # 将该状态-动作计数设为1
                else:
                    # 否则说明其他幕中出现过该状态-动作，并且曾经统计到final_state_action_reward_dict和final_state_action_count_dict变量里面
                    # 直接累加就好了。
                    final_state_action_reward_dict[state_action] += reward
                    final_state_action_count_dict[state_action] += 1

            if episode % 100 == 0:
                print("第{}个episode已完成=====已花费{}分钟".format(episode, (time.time() - begin_time) / 60))

        # 计算下最终的状态-动作价值
        # 由于是按概率采样，因此可能会导致某些动作-状态没有出现过，这个时候就需要一些方法去解决了。
        # 一种方法是增加采样次数，这种方法相当于是暴力解决。
        # 另一种方法可以参考sutton的《强化学习第二版》的98页的5.4内容
        self.averaged_state_action_value_dict = {}
        for state_action, reward in final_state_action_reward_dict.items():
            self.averaged_state_action_value_dict[state_action] = reward / final_state_action_count_dict[state_action]

        # print(self.averaged_state_action_value_dict)


    def show_policy(self):
        policy_dict = {}
        for state_action, value in self.averaged_state_action_value_dict.items():
            if state_action[0:2] not in policy_dict.keys():
                policy_dict[state_action[0:2]] = {self.action_dict[int(state_action[2])]: value}
            else:
                policy_dict[state_action[0:2]][self.action_dict[int(state_action[2])]] = value

        print(policy_dict)


obj = First_Visit_Monte_Carlo_Policy_Evaluation()
obj.fire_calculation()
obj.show_policy()