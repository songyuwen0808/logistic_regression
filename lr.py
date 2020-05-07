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

# for循环 + BP版本
# 特征数量
feature_num = len(train_info) - 1
# 样本数量
data_num = len(train_info[0])

# LR的初始化不用使用高斯随机
w = [0] * feature_num
b = 0
rate = 0.001

cycle_num = 100000
for k in range(cycle_num):
    # 训练cycle_num次
    z = [0] * data_num
    a = [0] * data_num
    l = [0] * data_num
    j = 0
    
    da = [0] * data_num
    dz = [0] * data_num
    dw = [0] * feature_num
    db = 0
    
    for i in range(data_num):
        # 计算z = w^T * x + b
        for j in range(feature_num):
            z[i] += w[j] * train_info[j][i]
        z[i] += b

        # 计算sigmoid
        a[i] = 1 / (1 + math.exp(z[i] * -1))
        
        # 计算loss function = -(y * loga + (1 - y) * log(1 - a))
        l[i] = -1 * (train_info[8][i] * math.log(a[i]) + (1 - train_info[8][i]) * math.log(1 - a[i]))

        # 计算cost function = sum(loss function)
        j += l[i]
        
        # 计算da = - y / a + (1 - y) / (1 - a)
        da[i] = -1 * train_info[8][i] / a[i] + (1 - train_info[8][i]) / (1 - a[i])
        
        # 计算dz = a - y
        dz[i] = a[i] - train_info[8][i]
        
        # 计算dw = x * dz
        for j in range(feature_num):
            dw[j] += train_info[j][i] * dz[i]
        db += dz[i]
        
    j /= data_num
    for j in range(feature_num):
        dw[j] /= data_num
        w[j] -= rate * dw[j]
        
    db /= data_num
    b -= rate * db
    
    print(w[0], b)

# 验证结果
test_num = len(train_info[0])
test_z = [0] * len(train_info[0])
for i in range(test_num):
    for j in range(feature_num):
        test_z[i] += w[j] * train_info[j][i]
        
    test_z[i] += b
    test_z[i] = 1 / (1 + math.exp(test_z[i] * -1))
    print(test_z[i], train_info[8][i], test_z[i] > 0.5)

#numpy + bp版本

#numpy + 牛顿迭代版本
