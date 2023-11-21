
# def print_hi(name):
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np

# 定义两个矩阵
matrix_a = np.array([[1, 4, 7, 4, 1]])
matrix_b = np.array([[1], [4], [7], [4], [1]])

# 执行矩阵乘法
result_matrix = np.dot(matrix_b, matrix_a)

# 或者使用 @ 操作符
# result_matrix = matrix_a @ matrix_b

print("矩阵 A:")
print(matrix_a)

print("矩阵 B:")
print(matrix_b)

print("矩阵乘法的结果:")
print(result_matrix)
