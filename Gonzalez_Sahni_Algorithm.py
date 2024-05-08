def open_shop_scheduling(pro_times):
    # 两台机器，n个工件的开放车间调度问题
    n = len(pro_times)  # 工件数
    a0, b0 = 0, 0  # 定义一个虚拟工件
    T1, T2 = 0, 0  # 初始化每台机器上的加工时间加工时间
    l, r = 0, 0  # 调度计划中的leftmost和rightmost工件
    S1, S2 = [], []  # 每台机器的初始加工序列

    # 98年的论文，关键公式的下标看不清，故无法完成算法编写，麻了
    for i in range(n):
        ai, bi = pro_times[i][0], pro_times[i][1]  # 第i个工件在第0/1台机器上的加工时间
        # 更新每台机器上的总加工时间
        T1 += ai
        T2 += bi

        if ai >= bi:
            pass    # 从右边开始添加
        elif ai < bi:
            pass    # 从左边开始添加
        else:
            pass

    if T1 - pro_times[l][0] < T2 - pro_times[r][1]:
        S2.insert(0, l)
    else:
        S1.insert(0, r)

    S1.extend(S2)

    # Adjust job indices to start from 0
    S1 = [job - 1 for job in S1]

    return S1


if __name__ == '__main__':
    # Example usage with the given processing times
    pro_times = [
        [10, 7, 3, 1, 12, 6],
        [6, 9, 8, 2, 7, 6]
    ]

    r = open_shop_scheduling(pro_times)
    print("Optimal Schedule:", r)
