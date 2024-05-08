# @description: 带链式约束的总加权完成时间：只要机器空闲了，选择剩下链中具有最高ρ因子的链，无间断地加工该链直到并包括确定它的ρ因子的工件为止。
import random


def cal_rho(train):
    """
    计算ρ因子
    :param train: tuple, train = (ids, w,p), ids: list,工件编号, w: list,工件权重, p: list,工件加工时间
    :return rho: float, ρ因子
    :return rho_id: int, ρ因子决定工序
    """
    ids, w, p = train
    w_sum = 0
    p_sum = 0
    rho = 0
    rho_id = 0
    for i, id in enumerate(ids):
        w_sum += w[i]  # 工件累积权重
        p_sum += p[i]  # 工件累积加工时间
        wp_sum = w_sum / p_sum  # 工件累积权重/工件累积加工时间
        if wp_sum > rho:
            rho, rho_id = wp_sum, id
    return rho, rho_id


if __name__ == '__main__':
    ids = [1, 2, 3, 4, 5, 6, 7]
    w = [0, 18, 12, 8, 8, 17, 16]  # 工件权重
    p = [3, 6, 6, 5, 4, 8, 9]  # 工件加工时间
    choose_id = random.choice(ids)  # 不同的子链分割会影响工件的最后调度结果
    # choose_id = 5
    print(choose_id)
    train1 = (ids[:choose_id - 1], w[:choose_id - 1], p[:choose_id - 1])
    train2 = (ids[choose_id - 1:], w[choose_id - 1:], p[choose_id - 1:])
    rho1, rho_id1 = cal_rho(train1)
    rho2, rho_id2 = cal_rho(train2)

    res = []
    while len(res) < len(ids):
        if rho1 > rho2:
            res += train1[0][:rho_id1]
            train1 = (train1[0][rho_id1:], train1[1][rho_id1:], train1[2][rho_id1:])
        else:
            res += train2[0][:rho_id2]
            train2 = (train2[0][rho_id2:], train2[1][rho_id2:], train2[2][rho_id2:])
        rho1, rho_id1 = cal_rho(train1)
        rho2, rho_id2 = cal_rho(train2)
    print(res)
