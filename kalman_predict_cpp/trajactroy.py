# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import sympy
import random
 
# 卡尔曼滤波器
class KalmanFilter:
    B = 0  # 控制变量矩阵，初始化为0
    u = 0  # 状态控制向量，初始化为0
    K = float('nan')  # 卡尔曼增益无需初始化
    z = float('nan')  # 观测值无需初始化，由外界输入
    P = np.diag(np.ones(4))  # 先验估计协方差
 
    x = []  # 滤波器输出状态
    G = []  # 滤波器预测状态
 
    # 状态转移矩阵A，和线性系统的预测机制有关
    A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
 
    # 噪声协方差矩阵Q，代表对控制系统的信任程度，预测过程上叠加一个高斯噪声，若希望跟踪的轨迹更平滑，可以调小
    Q = np.diag(np.ones(4)) * 0.1
 
    # 观测矩阵H：z = H * x，这里的状态是（坐标x， 坐标y， 速度x， 速度y），观察值是（坐标x， 坐标y）
    H = np.eye(2, 4)
 
    # 观测噪声协方差矩阵R，代表对观测数据的信任程度，观测过程上存在一个高斯噪声，若观测结果中的值很准确，可以调小
    R = np.diag(np.ones(2)) * 0.1
 
    def init(self, px, py, vx, vy):
        # 本例中，状态x为（坐标x， 坐标y， 速度x， 速度y），观测值z为（坐标x， 坐标y）
        self.B = 0
        self.u = 0
        self.K = float('nan')
        self.z = float('nan')
        self.P = np.diag(np.ones(4))
        self.x = [px, py, vx, vy]
        self.G = [px, py, vx, vy]
        self.A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
        self.Q = np.diag(np.ones(4)) * 0.1
        self.H = np.eye(2, 4)
        self.R = np.diag(np.ones(2)) * 0.1
 
    def update(self):
        # Xk_ = Ak*Xk-1+Bk*Uk
        a1 = np.dot(self.A, self.x)
        a2 = self.B * self.u
        x_ = np.array(a1) + np.array(a2)
        self.G = x_
 
        # Pk_ = Ak*Pk-1*Ak'+Q
        b1 = np.dot(self.A, self.P)
        b2 = np.dot(b1, np.transpose(self.A))
        p_ = np.array(b2) + np.array(self.Q)
 
        # Kk = Pk_*Hk'/(Hk*Pk_*Hk'+R)
        c1 = np.dot(p_, np.transpose(self.H))
        c2 = np.dot(self.H, p_)
        c3 = np.dot(c2, np.transpose(self.H))
        c4 = np.array(c3) + np.array(self.R)
        c5 = np.linalg.matrix_power(c4, -1)
        self.K = np.dot(c1, c5)
 
        # Xk = Xk_+Kk(Zk-Hk*Xk_)
        d1 = np.dot(self.H, x_)
        d2 = np.array(self.z) - np.array(d1)
        d3 = np.dot(self.K, d2)
        self.x = np.array(x_) + np.array(d3)
 
        # Pk = Pk_-Kk*Hk*Pk_
        e1 = np.dot(self.K, self.H)
        e2 = np.dot(e1, p_)
        self.P = np.array(p_) - np.array(e2)
 
    def accuracy(self, predictions, labels):
        return np.array(predictions) / np.array(labels)
 
 
# 读取敌机飞行数据
path = './9.xlsx'
label = pd.read_excel(path, header=None)
label_x = list(label.iloc[::, 0])
label_y = list(label.iloc[::, 1])
label_data = np.array(list(zip(label_x, label_y)))
 
# 读取我方雷达对敌机的侦查数据
path = './10.xlsx'
detect = pd.read_excel(path, header=None)
detect_x = list(detect.iloc[::, 0])
detect_y = list(detect.iloc[::, 1])
detect_data = np.array(list(zip(detect_x, detect_y)))
 
# 创建卡尔曼滤波器
t = len(detect_data)  # 处理时刻
kf_data_filter = np.zeros((t, 4))  # 滤波数据
kf_data_predict = np.zeros((t, 4))  # 预测数据
kf = KalmanFilter()  # 创建滤波器
kf.init(detect_x[0], detect_y[0], 0, 0)  # 滤波器初始化
 
# 生成地图画布
fig, ax = plt.subplots(1, 1)
plt.grid(ls='--')
ax.set_xlim(600, 800)
ax.set_ylim(300, 700)
 
# 初始化信息
fly_data_x = [label_data[0][0], ]
fly_data_y = [label_data[0][1], ]
 
missile_data_x = [625, ]
missile_data_y = [350, ]
 
line_fly, = plt.plot(fly_data_x[0],fly_data_y[0], 'r-')
line_missile,  = plt.plot(missile_data_x[0], missile_data_y[0], 'g-')
 
hit_flag = 0
hit_frame = -1
trace_flag = 1
 
# 计算我方导弹下一次的移动坐标
def missile_move(loc):
    global hit_flag
    solve_x = 0
    solve_y = 0
    x1, y1, x2, y2 = loc
    dist = ((x1-x2)**2 + (y1-y2)**2)**(1/2)
    max_dist = max(0.08*dist, 10)
    move_dist = min(max_dist*(0.6+random.random()), max_dist)
    if abs(dist - move_dist) < 5:
        hit_flag = 1
    x, y = sympy.symbols("x y")
    res = sympy.solve(
        [(y2-y1)*(x-x1) - (y-y1)*(x2-x1),
        ((x-x1)**2 + (y-y1)**2)**(1/2) - move_dist],
        [x, y]
        )
    for i in range(len(res)):
        if res[i][0] > min(x1, x2) and res[i][0] < max(x1, x2):
            solve_x =res[i][0]
            solve_y =res[i][1]
            break
    else:
        solve_x = x1
        solve_y = y1
    return solve_x, solve_y
 
# 初始化敌机、我方导弹位置
def fly_init():
    line_fly.set_data(fly_data_x, fly_data_y)
    line_missile.set_data(missile_data_x, missile_data_y)
    return line_fly, line_missile,
 
# 刷新敌机、我方导弹实时运动轨迹
def fly_update(frames):
    global fly_data_x, fly_data_y, missile_data_x, missile_data_y
    global line_fly, line_missile
    global hit_flag, hit_frame, trace_flag
    if hit_flag:
        hit_flag = 0
        trace_flag = 0
        hit_frame = frames.copy()
        plt.cla()
        plt.grid(ls='--')
        ax.set_xlim(600, 800)
        ax.set_ylim(300, 700)
        line_fly, = plt.plot(label_data[frames-1][0], label_data[frames-1][1], 'b*')
        line_missile, = plt.plot(label_data[frames-1][0], label_data[frames-1][1], 'b*')
 
    if hit_frame >= 0 and (frames >= hit_frame + 1):
        hit_frame = -1
        trace_flag = 0
 
    if frames >= (len(label_data) - 1):
        trace_flag = 1
        fly_data_x = [label_data[0][0], ]
        fly_data_y = [label_data[0][1], ]
        missile_data_x = [625, ]
        missile_data_y = [350, ]
        plt.cla()
        plt.grid(ls='--')
        ax.set_xlim(600, 800)
        ax.set_ylim(300, 700)
        line_fly, = plt.plot(fly_data_x[0],fly_data_y[0], 'r-')
        line_missile,  = plt.plot(missile_data_x[0], missile_data_y[0], 'g-')
    else:
        if trace_flag:
            fly_data_x.append(label_data[frames][0])
            fly_data_y.append(label_data[frames][1])
            line_fly.set_data(fly_data_x, fly_data_y)
            # ------关键处理步骤------
            kf.z = np.transpose([detect_x[frames], detect_y[frames]])  # 获取最新的观测数据
            kf.update()  # 更新卡尔曼滤波器参数
            kf_data_filter[frames, ::] = np.transpose(kf.x) # 滤波器输出
            loc = missile_data_x[frames-1], missile_data_y[frames-1],\
                    kf_data_filter[frames][0], kf_data_filter[frames][1]
            # ------关键处理步骤------
            move_x, move_y = missile_move(loc)
            missile_data_x.append(move_x)
            missile_data_y.append(move_y)
            line_missile.set_data(missile_data_x, missile_data_y)
    return line_fly, line_missile,
 
fly_anim = animation.FuncAnimation(fig=fig, func=fly_update,
                                frames=np.arange(1, len(label_data)),
                                init_func=fly_init, interval=100, blit=True)
 
 
plt.title('kalman filter trace object')
legend = ['fly', 'missile']
plt.legend(legend, loc="best", frameon=False)
fly_anim.save('animation.gif', writer='pillow', fps=10)
plt.show()