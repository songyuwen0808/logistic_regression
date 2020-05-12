import numpy
import math

# 使用西瓜书中的特征样本
# 色泽
color_map = {
    '青绿' : 1,
    '乌黑' : 2,
    '浅白' : 3
}
# 根底
root_map = {
    '蜷缩' : 1,
    '稍蜷' : 2,
    '硬挺' : 3
}
# 敲声
sound_map = {
    '浊响' : 1,
    '沉闷' : 2,
    '清脆' : 3
}
# 纹理
texture_map = {
    '清晰' : 1,
    '稍糊' : 2,
    '模糊' : 3
}
# 脐部
umbilical_map = {
    '凹陷' : 1,
    '稍凹' : 2,
    '平坦' : 3
}
# 触感
touch_map = {
    '硬滑' : 1,
    '软粘' : 2
}

# 分类类型
good_type = {
    '好瓜' : 1,
    '坏瓜' : 0
}


# https://blog.csdn.net/songyuwen0808/article/details/105378072
train_info = [
    ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白', '青绿', '浅白', '青绿'],
    ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷', '蜷缩', '蜷缩', '蜷缩'],
    ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '清脆', '清脆', '浊响', '浊响', '沉闷', '浊响', '浊响', '沉闷'],
    ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '模糊', '模糊', '稍糊', '稍糊', '清晰', '模糊', '稍糊'],
    ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '平坦', '平坦', '平坦', '凹陷', '凹陷', '凹陷', '平坦', '稍凹'],
    ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑', '硬滑', '硬滑', '硬滑'],
    [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.437, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
    [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.211, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103],
    ['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜']
]

test_info = [
    ['乌黑', '乌黑'],
    ['稍蜷', '稍蜷'],
    ['沉闷', '浊响'],
    ['稍糊', '稍糊'],
    ['稍凹', '稍凹'],
    ['硬滑', '软粘'],
    [0.666, 0.481],
    [0.091, 0.149],
    ['坏瓜', '好瓜'],
]

def transfer_info(data_info):
    for idx in range(len(data_info[0])):
        data_info[0][idx] = color_map[data_info[0][idx]]
        data_info[1][idx] = root_map[data_info[1][idx]]
        data_info[2][idx] = sound_map[data_info[2][idx]]
        data_info[3][idx] = texture_map[data_info[3][idx]]
        data_info[4][idx] = umbilical_map[data_info[4][idx]]
        data_info[5][idx] = touch_map[data_info[5][idx]]
        data_info[8][idx] = good_type[data_info[8][idx]]        

print("=================训练集转换前=================")
for line in train_info:
    print(line)
    
transfer_info(train_info)

print("=================训练集转换前=================")
for line in train_info:
    print(line)


print("=================测试集转换前=================")
for line in test_info:
    print(line)

transfer_info(test_info)
print("=================测试集转换前=================")
for line in train_info:
    print(line)
cycle_num = 100000
rate = 0.01


# 从隐层开始,各个层的节点个数
node_num = [4, 1]
feature_num = len(train_info) - 1
data_num = len(train_info[0])
# 特征向量
X = np.array(train_info[:-1])
Y = np.array(train_info[-1])
W1 = np.random.randn(node_num[0], len(X)) * 0.01
B1 = np.random.randn(node_num[0], 1) * 0.01
W2 = np.random.randn(node_num[1], node_num[0]) * 0.01
B2 = np.random.randn(node_num[1], 1) * 0.01
print("W1 = ", W1)
print("B1 = ", B1)
print("W2 = ", W2)
print("B2 = ", B2)
print("X = ", X)
print("Y = ", Y)

# 开始训练
for _ in range(cycle_num):
    Z1 = np.dot(W1, X) + B1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + B2
    A2 = 1 / (1 + np.exp(-Z2))
    dz2 = A2 - Y
    dw2 = np.dot(dz2, A1.T) / data_num
    db2 = np.sum(dz2, axis = 1, keepdims = True) / data_num
    dz1 = np.dot(W2.T, dz2) * A1 * (1 - A1)
    dw1 = np.dot(dz1, X.T) / data_num
    db1 = np.sum(dz1, axis = 1, keepdims = True) / data_num
    W1 = W1 - rate * dw1
    W2 = W2 - rate * dw2
    B1 = B1 - rate * db1
    B2 = B2 - rate * db2
    
test_z1 = np.dot(W1, X) + B1
test_a1 = 1 / (1 + np.exp(-test_z1))
test_z2 = np.dot(W2, test_a1) + B2
test_a2 = 1 / (1 + np.exp(-test_z2))
print(test_a2)
