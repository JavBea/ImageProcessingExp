import numpy as np
from scipy.optimize import fsolve


# 定义映射函数
def mapping_equation(vars, x_target, y_target):
    x, y = vars
    r = np.sqrt(x ** 2 + y ** 2)
    theta = (1 - r) ** 2
    if r >= 1:
        return [x - x_target, y - y_target]
    else:
        x_prime = np.cos(theta) * x - np.sin(theta) * y
        y_prime = np.sin(theta) * x + np.cos(theta) * y
        return [x_prime - x_target, y_prime - y_target]


def demapping(x, y):
    # 使用fsolve求解非线性方程组
    initial_guess = [0, 0]  # 初始猜测值
    solution = fsolve(lambda vars: mapping_equation(vars, x, y), initial_guess)
    print(solution)
    return solution


if __name__ == "__init__":
    demapping(0.2, 0.5)
    demapping(0.3, 0.8)
