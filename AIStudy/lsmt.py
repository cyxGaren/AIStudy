import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    data = pd.read_csv('dataset_1.csv',index_col=['time'])
    data.sort_index(inplace=True)
    # plt.xticks([])
    # plt.plot(data[1:],'-')
    # plt.show()

    print(data.shape,'1')
    data = np.array(data)
    print(data.shape,'1')
    normalize_data = (data - np.mean(data)) / np.std(data)


    time_step = 20  # 时间步
    rnn_unit = 10  # hidden layer units
    batch_size = 60  # 每一批次训练多少个样例
    input_size = 1  # 输入层维度
    output_size = 1  # 输出层维度
    lr = 0.0006  # 学习率
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalize_data) - time_step - 1):
        x = normalize_data[i:i + time_step]
        y = normalize_data[i + 1:i + time_step + 1]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    print(train_x)


load_data()