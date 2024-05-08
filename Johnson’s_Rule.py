# 适用条件：m=2，有序，即每一工件都必须先经过机器1，再经过机器2，进行加工
def get_min_time(pro_times):
    if not pro_times or not pro_times[0]:
        return None, None

    min_time = float('inf')
    min_index = (0, 0)
    for i in range(len(pro_times)):
        for j in range(len(pro_times[i])):
            if pro_times[i][j] < min_time:
                min_time = pro_times[i][j]
                min_index = (i, j)
    return min_time, min_index


def johnson_rule(pro_times):
    res = [[], []]
    while True:
        min_time, min_index = get_min_time(pro_times)   # 将𝐴𝑖与𝐵𝑖数值分列为两列，找到两列中最小的值
        i, j = min_index
        if min_time != float('inf'):
            if i == 0:  # 如果最小数值出现在a列中，将对应工件排在前面
                res[0].append(j)
            else:   # 如果最小数值出现在B列中，则将对应工件排在后面

                res[1].insert(0, min_index[1])
            pro_times[i][j] = float('inf')  # 将已安排完的工件划掉，继续上述过程，直至所有工件都排完
        else:
            break
    return res


if __name__ == '__main__':
    # 机器编号为0,1，工件编号为0~4，均为二维数组的下标
    pro_times = [[5, 1, 9, 3, 10],
                 [2, 6, 7, 8, 4]]

    r = johnson_rule(pro_times)
    print(r)
