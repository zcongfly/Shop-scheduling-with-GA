# 适用条件：m=2
# 步骤：1.确定具有最长加工时长的操作，安排其在该机器上最后加工，在另一台机器上优先加工
# 2. 剩余的操作只需要满足没有机器空闲即可
# 书上写的原话，说实话觉得问题很大，但是有找不到更好的资料，先这样，后期再改
def get_max_time(pro_times):
    if not pro_times or not pro_times[0]:
        return None, None

    max_time = -1
    max_index = (0, 0)
    for i in range(len(pro_times)):
        for j in range(len(pro_times[i])):
            if pro_times[i][j] > max_time:
                max_time = pro_times[i][j]
                max_index = (i, j)
    return max_time, max_index


def johnson_rule(pro_times):
    if not pro_times or not pro_times[0]:
        return None

    n = len(pro_times[0])
    res = [[None for _ in range(n)], [None for _ in range(n)]]
    la, lb = 0, 0
    ra, rb = n - 1, n - 1
    while True:
        max_time, max_index = get_max_time(pro_times)
        i, j = max_index
        if max_time != 0:
            if i == 0:  # 如果最大数值出现在a列中，将对应工件排在a列的最后面，同时将其排在b列的最前面
                res[0][ra] = j
                res[1][lb] = j
                ra -= 1
                lb += 1
            else:
                res[1][rb] = j
                res[0][la] = j
                rb -= 1
                la += 1
            pro_times[0][j] = 0
            pro_times[1][j] = 0
        else:
            break
    return res


if __name__ == '__main__':
    # 机器编号为0,1，工件编号为0~4，均为二维数组的下标
    pro_times = [[5, 1, 9, 3, 10],
                 [2, 6, 7, 8, 4]]

    r = johnson_rule(pro_times)
    print(r)
