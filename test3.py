def fitness(job):
    delay_times = [[] for _ in range(machine_num)]  # 每个工件超过交货期的延时时间
    finish_times = [[] for _ in range(machine_num)]  # 每个工件完成加工的时间点
    for i in range(len(job)):
        if finish_times[job[i]]:
            finish_times[job[i]].append(
                pro_times[job_id[i]] + max(finish_times[job[i]][-1], arr_times[job_id[i]]))
        else:
            finish_times[job[i]].append(pro_times[job_id[i]] + arr_times[job_id[i]])
        delay_times[job[i]].append(max(finish_times[job[i]][-1] - deadlines[job_id[i]], 0))

    return sum(element for sublist in delay_times for element in sublist)

if __name__ == '__main__':
    # 定义多机调度问题的工件和加工时间
    pro_times = [4, 7, 6, 5, 8, 3, 5, 5, 10]  # 加工时间
    arr_times = [3, 2, 4, 5, 3, 2, 1, 8, 6]  # 到达时间
    deadlines = [10, 15, 30, 24, 14, 13, 20, 18, 10]  # 交货期
    machine_num = 3  # 3台完全相同的并行机，编号为0,1,2


    job_id = [5, 4, 0, 1, 6, 3, 8, 2, 7]
    best_job = [0, 0, 1, 1, 0, 0, 1, 1, 0]
    best_makespan = fitness(best_job)
    print("最佳加工顺序：", job_id)
    print("最佳调度分配：", best_job)
    print("最小交货期延时时间：", best_makespan)
