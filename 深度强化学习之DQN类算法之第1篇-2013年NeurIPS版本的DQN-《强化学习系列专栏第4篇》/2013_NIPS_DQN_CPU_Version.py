# encoding=utf-8
'''
Author: Haitaifantuan
Create Date: 2020-09-27 23:23:52
Author Email: 47970915@qq.com
Description: Should you have any question, do not hesitate to contact me via E-mail.
'''
import gym
import torch.nn as nn
import torch
from torchvision import transforms
import atari_py
import random
import time
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import copy
import os


class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

        self.preprocessing = transforms.Compose([
            # 按照论文步骤
            # 先转换为灰度图
            transforms.Grayscale(1),
            # 再下采样到110*84的大小
            transforms.Resize((110, 84)),
            # 转换为Tensor()输入到网络
            transforms.ToTensor()
        ]
        )

    def forward(self, input):
        # 由于传进来是torch.Tensor()
        # 所以我们要将其转换为PIL.Image才能预处理
        input = Image.fromarray(input)
        # 最后输出的就是论文所说的84*84的灰度图像了
        output = self.preprocessing(input)  # 这个时候output是[1, 84, 84]
        # 将多余的维度压缩掉，最后返回的是[84, 84]的形状
        output = torch.squeeze(output)
        # 然后再裁剪到84*84的大小的游戏区域
        output = output[17:101, :]  # 这个区域是游戏的区域
        # plt.imshow(output, cmap='gray')
        # plt.show()
        return output


class Deep_Q_Network(nn.Module):
    def __init__(self, action_nums):
        super(Deep_Q_Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, (8, 8), 4),
            nn.ReLU(),  # 论文中使用的不一定是这个激活函数，这里是为了简化使用ReLU
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU()  # 论文中使用的不一定是这个激活函数，这里是为了简化使用ReLU
        )

        self.classifier = nn.Sequential(
            nn.Linear(2592, 256),
            nn.Linear(256, action_nums)
        )

    def forward(self, input):
        output = self.features(input)
        output = output.view(-1, 2592)
        output = self.classifier(output)
        output = torch.squeeze(output)
        return output

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)


class Agent(object):
    def __init__(self):
        # 模型保存的路径
        self.model_path = './2013_NIPS_DQN_cpu_trained_model_save_reward_loss/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.save_model_path = self.model_path + '/model'

        self.lr = 0.001

        # 我们玩“乒乓球游戏”，这里搭建下环境
        self.env = gym.make('Pong-v0')
        self.env = self.env.unwrapped
        # 这个是游戏的valid的动作
        self.action_space = self.env.action_space.n
        self.action_nums = self.env.action_space.n
        # 构建图像预处理对象
        self.preprocessing = Preprocessing()
        # 构建deep q-network网络
        self.deep_q_network = Deep_Q_Network(self.action_nums)
        # 初始化网络
        if os.path.exists(self.save_model_path):
            state_dict = torch.load(self.save_model_path)
            self.deep_q_network.load_state_dict(state_dict)
            print("从已训练好的模型中加载模型成功")
        else:
            self.deep_q_network.initialization()
            print("初始化模型所有参数成功")
        # 构建损失函数
        self.loss_func = nn.MSELoss()
        # 构建优化器
        self.opti = torch.optim.SGD(self.deep_q_network.parameters(), lr=self.lr, momentum=0.9)
        # 每次训练的样本数量，论文中是32
        self.batch_size = 32
        # 创建一个缓存，超过大小后，最新的放进去，老的扔掉
        self.replay_memory_size = 200000  # 30000的话，2080Ti显存11G不够  10万需要20个G内存左右
        self.replay_memory = deque()
        # 当memory_size达到多少后，开始训练
        self.begin_to_train_memory_size = 50000

        self.alpha = 0.9
        self.gamma = 0.9
        self.init_epsilon = 1  # 论文为1
        self.final_epsilon = 0.1  # 论文为0.1
        self.epsilon_decay_frames = 1000000  # 论文1000000

        self.train_times = 0

        # 论文中是每4帧，agent进行一次动作的选择。
        self.select_action_every_k_time = 4

        # 记录reward变化的变量
        self.reward_change = []
        # 记录loss变化的变量
        self.loss_change = []

    def four_img_list_to_Q_net_input(self, four_img_list):
        stacked = torch.stack(list(four_img_list))
        return stacked

    def generate_initial_4_frames(self, current_state_single_frame):
        '''
        由于环境一开始，four_img_list的长度是小于4的
        因此我们需要让其长度达到4后，再继续后面的记录操作
        在前4步，我们都使用模型选择动作
        :param current_state_single_frame:
        :return: 返回一个队列，里面存放了第1、2、3、4帧游戏画面的对应的Tensor数值
        '''
        four_img_list = deque()
        # 由于一开始并没有4张图片可以使用
        # 因此，我们根据当前的状态，复制出另外3张图片
        # 然后随着step的进行，我们一张图片一张图片的放进去
        four_img_list.extend([current_state_single_frame, current_state_single_frame,
                              current_state_single_frame, current_state_single_frame])

        for _ in range(3):
            # 渲染环境
            self.env.render()
            # 这里将4帧图片变成4个通道放到网络里
            current_state_4_frames_stacked_result = self.four_img_list_to_Q_net_input(four_img_list)
            # 放到网络里需要再添加一个Batch_size部分的维度
            current_state_4_frames_stacked_result = torch.unsqueeze(current_state_4_frames_stacked_result, dim=0)
            action_value = self.deep_q_network(current_state_4_frames_stacked_result)
            action = torch.argmax(action_value)
            next_state, reward, done, info = self.env.step(action)
            next_state_to_tensor = self.preprocessing(next_state)

            four_img_list.append(next_state_to_tensor)
            four_img_list.popleft()

        return four_img_list

    def train(self):
        # 原始论文：如果达到了replay_memory的最大值，那就开始从replay_memory中随机选取样本进行训练
        # if len(self.replay_memory) > (self.replay_memory_size - 1):
        if len(self.replay_memory) > self.begin_to_train_memory_size:
            batch_data = random.choices(self.replay_memory, k=32)

            # 拿到训练数据后，将他们进行解包
            current_state_4_frames_stacked_result_list = [each[0] for each in batch_data]
            current_state_action_list = torch.LongTensor([[each[1]] for each in batch_data])
            reward_list = torch.FloatTensor([[each[2]] for each in batch_data])
            next_state_4_frames_stacked_result_list = [each[3] for each in batch_data]
            done_list = [[each[4]] for each in batch_data]

            # 将训练数据放到模型里进行前向传播
            y_pre = self.deep_q_network(torch.stack(current_state_4_frames_stacked_result_list).squeeze()).gather(dim=1,
                        index=current_state_action_list)

            # 根据公式，构建标签值
            q_net_result = self.deep_q_network(torch.stack(next_state_4_frames_stacked_result_list, dim=0)).detach()
            y_target = reward_list + self.gamma * torch.max(q_net_result, dim=1)[0].reshape(self.batch_size, -1)

            self.loss = self.loss_func(y_pre, y_target)

            self.opti.zero_grad()
            self.loss.backward()
            self.opti.step()
            self.train_times += 1

    def close_env(self):
        self.env.close()

    def save_model(self):
        torch.save(self.deep_q_network.state_dict(), self.save_model_path)

    def fire_in_the_hole(self):
        frame_count = 0
        self.current_epsilon = self.init_epsilon
        self.begin_time = time.time()
        for self.episode in range(100000):
            # 一个episode结束后，重新设置下环境，返回到随机的一个初始状态
            current_state_single_frame = self.env.reset()
            # 将current_state()预处理一下然后转换为Tensor
            current_state = self.preprocessing(current_state_single_frame)

            # 这个方法返回的four_img_list里面就存放了第1、2、3、4帧画面的Tensor()形式
            four_img_list = self.generate_initial_4_frames(current_state)
            current_state_4_frames_stacked_result = self.four_img_list_to_Q_net_input(four_img_list)

            # 记录一下当前这一盘总的reward
            self.current_episode_reward = 0
            self.select_action_count = 0
            while True:
                # 渲染环境
                self.env.render()

                # 论文每4帧才根据ε-greedy方法做一个动作
                # 其他3帧时间的动作选取上一轮选择的动作
                if self.select_action_count == 0 or self.select_action_count == self.select_action_every_k_time:
                    # 根据ε-greedy方法，走一步，看看
                    if random.random() < self.current_epsilon:
                        current_state_action = self.env.action_space.sample()
                    else:
                        # 根据Q函数找到最优的动作
                        # 放到网络里需要再添加一个Batch_size部分的维度
                        action_value = self.deep_q_network(
                            torch.unsqueeze(current_state_4_frames_stacked_result, dim=0))
                        current_state_action = torch.argmax(action_value)

                    self.select_action_count = 0

                next_state, reward, done, info = self.env.step(current_state_action)
                next_state_to_tensor = self.preprocessing(next_state)

                self.current_episode_reward += reward

                four_img_list.append(next_state_to_tensor)
                four_img_list.popleft()

                next_state_4_frames_stacked_result = self.four_img_list_to_Q_net_input(four_img_list)

                # （将当前的状态以及前三幅图片组成的图片，当前的行为，当前获得的奖励，下一个状态，游戏是否结束）添加到replay_memory中
                self.replay_memory.append((current_state_4_frames_stacked_result, current_state_action,
                                           reward, next_state_4_frames_stacked_result, done))
                if len(self.replay_memory) > self.replay_memory_size:
                    self.replay_memory.popleft()

                # 判断当前这一盘游戏是否结束
                if done:
                    self.end_time = time.time()
                    self.minute = int((self.end_time - self.begin_time) / 60)
                    self.hour = int(self.minute / 60)
                    self.day = int(self.hour / 24)

                    if len(self.replay_memory) < self.begin_to_train_memory_size:
                        self.loss = torch.tensor(0)
                    break

                current_state_4_frames_stacked_result = next_state_4_frames_stacked_result
                self.select_action_count += 1

                frame_count += 1
                if frame_count <= self.epsilon_decay_frames:
                    self.current_epsilon = self.init_epsilon - (
                            self.init_epsilon - self.final_epsilon) * frame_count / self.epsilon_decay_frames

                # 执行训练网络的操作，里面会判断reply_memory的长度是否达到最大值了
                self.train()

            self.reward_change.append(self.current_episode_reward)
            self.loss_change.append(self.loss.data.item())

            print(
                "当前已训练{}天-{}小时-{}分钟===当前为第{}个Episode===当前episode共获得{}reward===总共已训练{}次===当前的loss为\
                ：{}===当前的epsilon值为：{}===当前reply_memory的长度为：{}".format(
                    self.day, self.hour, self.minute, self.episode, self.current_episode_reward, self.train_times,
                    self.loss,
                    self.current_epsilon, len(self.replay_memory)))

            if self.episode % 10 == 0:
                # 保存模型
                self.save_model()
                # 将当前的self.reward_change列表保存下来，以覆盖的方式保存下来。
                with open(self.model_path + '/reward_change.txt', 'w', encoding='utf-8') as file:
                    file.write(str(self.reward_change))

                # 将当前的self.loss_change保存下来
                with open(self.model_path + '/loss_change.txt', 'w', encoding='utf-8') as file:
                    file.write(str(self.loss_change))

        # 关闭游戏环境
        self.close_env()


agent = Agent()
agent.fire_in_the_hole()
