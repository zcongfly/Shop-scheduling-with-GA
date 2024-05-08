import random
from matplotlib import pyplot as plt


class Individual:
    def __init__(self, processing_times, n, m):
        # self.code = [i for i in range(n * m)]   # 染色体编码
        # random.shuffle(self.code)   # 乱序
        self.code = [8, 6, 10, 5, 1, 7, 14, 3, 11, 2, 0, 4, 13, 15, 9, 12]
        self.decode, self.fitness = self.calc_fitness(processing_times, n, m)

    def check_job(self, start, limit, duration, job_activity):
        if len(job_activity) == 0:
            if start + duration <= limit:
                return True, start
        elif len(job_activity) == 1:
            if job_activity[0][0] != 0 and job_activity[0][0] >= start + duration:
                return True, start
            else:
                if job_activity[0][1] + duration <= limit:
                    return True, max(job_activity[0][1], start)
        else:
            end = start + duration
            j = 0
            while end <= limit and j < len(job_activity) - 1:
                if job_activity[j][0] <= start < job_activity[j][1]:
                    start = job_activity[j][1]
                    end = start + duration

                if start >= job_activity[j][1] and end <= job_activity[j + 1][0]:
                    return True, start

                j += 1

        return False, -1

    def calc_fitness(self, processing_times, n, m):
        jobs = [[] for _ in range(n)]  # 工件
        machines = [[] for _ in range(m)]  # 机器
        solution = [[] for _ in range(m)]  # 封装一个有效的调度方案
        for operation in self.code:
            machine = operation % n  # 根据操作编号计算所属的机器编号
            job = operation // n  # 根据操作编号计算所属的工件编号
            duration = processing_times[job][machine]  # 定位加工时间
            machine_activity = machines[machine]  # 当前时刻当前机器是否正在加工
            job_activity = jobs[job]  # 当前时刻当前工件是否正在加工
            # print(f'==========operation={operation}===========')
            # print("job_activity=",job_activity)
            # print("machine_activity=",machine_activity)

            fit, start, limit = False, -1, float('inf')
            if len(machine_activity) == 0:  # 如果机器空闲
                start = 0
                fit, start = self.check_job(start, limit, duration, job_activity)
            elif len(machine_activity) == 1 and machine_activity[0][
                0] != 0:  # 如果该机器只有一个操作，且该操作开始时间不为0，则判断一下当前操作是否可以排在该操作的前面
                start = 0
                limit = machine_activity[0][0]
                fit, start = self.check_job(start, limit, duration, job_activity)
            elif len(machine_activity) == 1 and machine_activity[0][0] == 0:  # 如果该机器只有一个操作，且该操作开始时间为0，则将当前操作排在该操作后边
                start = machine_activity[0][1]
                fit, start = self.check_job(start, limit, duration, job_activity)
            else:  # 如果该机器有两个或两个以上的操作，则看下操作与操作之间是否存在足够的间隙可容纳当前操作
                for i in range(len(machine_activity) - 1):
                    start = machine_activity[i][1]
                    limit = machine_activity[i + 1][0]
                    options = []
                    if limit - start >= duration:
                        fit, start = self.check_job(start, limit, duration, job_activity)
                        if fit:
                            options.append((start, limit - (start + duration)))
                    if len(options) > 0:
                        options.sort(key=lambda x: x[1])
                        start = options[0][0]
                        fit = True

            if fit and start + duration <= limit:
                jobs[job].append((start, start + duration))  # (开始时间,结束时间)
                machines[machine].append((start, start + duration))
                solution[machine].append((start, start + duration, operation))
                jobs[job].sort()  # 升序排序（时间是从小到大排列）
                machines[machine].sort()
                solution[machine].sort()
            else:
                start = 0
                if len(machine_activity) >= 1:
                    start = machine_activity[-1][1]
                if len(job_activity) >= 1:
                    start = max(start, job_activity[-1][1])
                jobs[job].append((start, start + duration))
                machines[machine].append((start, start + duration))
                solution[machine].append((start, start + duration, operation))

        print(jobs)
        print(machines)
        print(solution)

        for machine in machines:
            machine.sort(key=lambda x: x[1], reverse=True)

        return solution, max([machine[0][1] for machine in machines])


def plot_gantt(job_num, machine_num, solution):
    # 准备一系列颜色
    colors = ['blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta', 'SlateBlue',
              'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue', 'navajowhite',
              'moccasin', 'white', 'navy', 'sandybrown', 'moccasin']
    # colors = ['r', 'g', 'b', 'm', 'c', 'y', 'w', 'o', 'p']
    job_colors = random.sample(colors, job_num)

    start_times = [[-1 for _ in range(len(solution[0]))] for _ in range(len(solution))]
    end_times = [[-1 for _ in range(len(solution[0]))] for _ in range(len(solution))]
    job_decodes = [[-1 for _ in range(len(solution[0]))] for _ in range(len(solution))]
    for i in range(len(solution)):
        for j in range(len(solution[0])):
            start_times[i][j] = solution[i][j][0]
            end_times[i][j] = solution[i][j][1]
            job_decodes[i][j] = (solution[i][j][2] // machine_num, solution[i][j][2] % machine_num)

    # 创建图表和子图
    plt.figure(figsize=(12, 6))

    # 绘制工序的甘特图
    for i in range(len(start_times)):
        for j in range(len(start_times[i])):
            plt.barh(i, end_times[i][j] - start_times[i][j], height=0.5, left=start_times[i][j], color=job_colors[i][j],
                     edgecolor='black')
            plt.text(x=(start_times[i][j] + end_times[i][j]) / 2, y=i, s=job_decodes[i][j], fontsize=14)

    # 设置纵坐标轴刻度为机器编号
    machines = [f'Machine {i}' for i in range(len(start_times))]
    plt.yticks(range(len(machines)), machines)

    # 设置横坐标轴刻度为时间
    # start = min([min(row) for row in start_times])
    start = 0
    end = max([max(row) for row in end_times])
    plt.xticks(range(start, end + 1))
    plt.xlabel('Time')

    # 图表样式设置
    plt.ylabel('Machines')
    plt.title('Gantt Chart')
    # plt.grid(axis='x')

    # 自动调整图表布局
    plt.tight_layout()

    # 显示图表
    plt.show()


n, m = 4, 4
processing_times = [[34, 2, 54, 61],
                    [15, 89, 70, 9],
                    [38, 19, 28, 87],
                    [95, 7, 34, 29]]
# ub, lb = 193, 186
individual = Individual(processing_times, n, m)
print(individual.code)
print(individual.fitness)
plot_gantt(n, m, individual.decode)
