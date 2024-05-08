import matplotlib.pyplot as plt
import numpy as np


def Gantt(self):  # 预调度甘特图
    M = ['blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite', 'moccasin', 'white', 'white', 'white',
         'navy', 'sandybrown', 'moccasin', 'black']

    def get_name(k):
        if k == 18:
            return 'pm'
        elif k == 19:
            return '①'
        elif k == 20:
            return '②'
        elif k == 21:
            return '③'
        else:
            return k

    def get_x(k, i):
        if k == 18:
            return Start_time[i_1] + (End_time[i_1] - Start_time[i_1]) / 2 - 0.7
        else:
            return Start_time[i_1] + (End_time[i_1] - Start_time[i_1]) / 2 - 0.45

    for i in range(len(self.Machines)):
        Machine = self.Machines[i]
        Start_time = Machine.start
        End_time = Machine.end

        for i_1 in range(len(End_time)):
            # 彩色标签
            plt.barh(i, width=End_time[i_1] - Start_time[i_1], height=0.8, left=Start_time[i_1],
                     color=M[Machine._on[i_1][0]], edgecolor='black')
            # 开始结束时间 标签
            # plt.text(x=Start_time[i_1]+0.1,y=i,s=Machine.assigned_task[i_1])

            # 黑白色
            # plt.barh(i, width=End_time[i_1] - Start_time[i_1], height=0.8, left=Start_time[i_1], \
            #         color='white', edgecolor='black')
            # 标签加入，机器号
            plt.text(x=get_x(Machine._on[i_1][0], i), y=i,
                     s=get_name(Machine._on[i_1][0]), fontsize=13)
    plt.yticks(np.arange(i + 1), np.arange(1, i + 2))
    plt.title('Scheduling Gantt chart', fontsize=20)
    plt.ylabel('Machines', fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.gcf().set_size_inches(20, 8)
    plt.show()


class Machine:
    def __init__(self, start, end, on):
        self.start = start
        self.end = end
        self._on = on


# 模拟一个机器的任务安排
machine1 = Machine([0, 3, 6], [2, 4, 8], [1, 2, 3])
machine2 = Machine([1, 4, 7], [3, 6, 9], [4, 5, 6])
machine3 = Machine([2, 5, 8], [4, 7, 10], [7, 8, 9])

# 创建机器列表
machines = [machine1, machine2, machine3]

# 调用 Gantt 函数绘制甘特图
Gantt(machines)
