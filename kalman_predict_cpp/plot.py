import matplotlib.pyplot as plt

# 读取数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            # 将每行分割成独立的值
            values = line.split()
            # 将字符串转换为浮点数
            data.append([float(val) for val in values])
    return data

# 绘图
def plot_data(data):
    for i, column in enumerate(zip(*data)):
        plt.plot(column, label=f'Column {i+1}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Data Plot from TXT File')
    plt.legend()
    plt.show()

# 文件路径 - 需要根据您的文件位置进行更改
file_path = '/home/zxy/kalman/result.txt'

# 读取并绘图
data = read_data(file_path)
plot_data(data)
