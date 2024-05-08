# MOR（most operation remaining）规则：优先选择剩余工序数最多的工件进行加工
# SPT（shortest processing time）规则：优先选择加工时间最短的工序对应的工件进行加工
# 每个工件的3个工序在各机器上的加工顺序
machine_seq = [[2, 1, 0],
               [1, 0, 2],
               [0, 2, 1]]
# 每个工件的三道工序在各个机器上的加工时间
pro_times = [[3, 12, 5],
             [4, 8, 6],
             [2, 3, 7]]
machine_num = 3

choices = {0: 0, 1: 0, 2: 0}  # 工序候选集，对应三个工件的第0个工序
ch = -1
start_times = [[] for _ in range(machine_num)]   # 机器数m×工件数n
finish_times = [[] for _ in range(machine_num)]
while min(choices.values()) != machine_num:
    if max(choices.values()) == min(choices.values()):  # 如果三个工件的待完成工序相等，采用SPT规则
        min_time = float('inf')
        for i in range(len(choices)):
            if min_time > pro_times[i][choices[i]]:
                min_time = pro_times[i][choices[i]]
                ch = i
    else:  # 如果三个工件待完成的工序不相等，采用MOR规则
        min_seq = min(choices.values())
        min_time = float('inf')
        for i in range(len(choices)):
            if choices[i] == min_seq and min_time > pro_times[i][choices[i]]:
                min_time = pro_times[i][choices[i]]
                ch = i
    print(ch, choices[ch])   # 工件，机器
    choices[ch] += 1
