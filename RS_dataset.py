#coding=utf-8
import random
import math
import tqdm
import matplotlib.pyplot as plt
import time


#***********************************************************************
#@函数名称:               plot
#@功能描述:               将所有embedding按访问次数递增排序后，生成访问次数的累加图
#@参数:                  count_list      长度为embedding的个数,第i个代表编号为i的embedding被访问的次数
#                       embedding_num   数据集embedding的个数
#                       r               热数据占所有数据的比例
#@返回:                  none
#***********************************************************************
def plot(count_list, embedding_num, r):
    count_list_sorted = count_list.copy()
    count_list_sorted.sort()
    x = range(embedding_num)
    accumulation = 0
    y = []
    for i in range(embedding_num):
        accumulation += count_list_sorted[i]
        y.append(accumulation)
    plt.plot(x, y)
    #冷热数据的临界点
    plt.plot(x[int(embedding_num * (1-r))], y[int(embedding_num * (1-r))], marker='o')
    plt.show()


#***********************************************************************
#@函数名称:               table_creat
#@功能描述:               随机生成table，生成一个list，用于储存每个table的embedding数量
#@参数:                  embedding_num    数据集embedding的个数
#                       table_num        生成数据集中embedding table的个数
#@返回:                  table            w一个储存每个table中embedding数量的列表
#***********************************************************************
def table_creat(embedding_num, table_num):
    size = embedding_num - int(embedding_num / (table_num*1.2)) * table_num
    a = random.sample(range(1,size),table_num-1)
    a.extend([0,size])
    a.sort()
    table = [a[i+1] - a[i] for i in range(table_num)]
    for i in range(table_num):
        table[i] += int(embedding_num / (table_num*1.2))
    print(table)
    return table


#***********************************************************************
#@函数名称:               embedding_access_create
#@功能描述:               按照冷热比率，生成稀疏数据的访问记录
#@参数:                  embedding_dimension  生成数据集中embedding的维度
#                       embedding_size       生成数据集embedding所占空间大小，单位GB
#                       r                    热数据占所有数据的比例(0-1)， 数量为r的热数据占1-r的访问
#                       table_num            生成数据集中embedding table的个数
#                       sample_num           生成数据条数
#@返回:                  table                一个储存每个table中embedding数量的列表
#                       count_list           列表，记录每个embedding被访问的次数
#                       embedding_num        生成数据集embedding的数量
#***********************************************************************
def embedding_access_create(embedding_dimension, embedding_size, r, table_num, sample_num):
    #计算embedding的数量
    embedding_num = int(embedding_size * math.pow(2,30) / 4 / embedding_dimension)
    #随机生成table，生成一个list，用于储存每个table的embedding数量
    table = table_creat(embedding_num, table_num)
    # 用于记录每个embedding data访问的次数
    count_list = [0] * embedding_num
    # 用于记录每个table中所有embedding data访问的次数总和
    table_counter = [0] * table_num
    # 用于记录每个embedding data所在的table编号
    table_index = [0] * embedding_num
    begin = 0
    for i in range(table_num):
        table_index[begin: begin + table[i]] = [i] * table[i]
        begin += table[i]
    
    #随机生成冷热数据的数据编号
    hot_data_list, cold_data_list = [],[]
    for i in range(embedding_num):
        a = random.randint(1,100)
        if a <= r*100:
                hot_data_list.append(i)
        else:
                cold_data_list.append(i)
    hot_data_num = len(hot_data_list)
    
    #生成热数据的访问记录
    print('start hot data')
    counter = 0
    pbar_hot = tqdm.tqdm(total = int(table_num * sample_num * (1-r)))
    while counter < int(table_num * sample_num * (1-r)):
        access = random.choice(hot_data_list)
        #table中所有embedding data访问的次数总和不能超过训练数据条数
        if table_counter[int(table_index[access])] + 1 > sample_num:
            continue
        table_counter[int(table_index[access])] += 1
        count_list[access] += 1
        counter += 1
        pbar_hot.update(1)
    
    #找出访问次数最少热数据的访问次数，冷数据的访问次数不能超过该值
    min_hot = count_list[hot_data_list[0]]
    for embedding in hot_data_list:
        min_hot = min((min_hot,count_list[embedding]))
    print('min_hot',min_hot)

    #计算每个table还要生成的冷数据访问次数
    cold_data_table_counter  = []
    cold_dic = {}
    for i in range(table_num):
        cold_data_table_counter.append(sample_num - table_counter[i])
        cold_dic[i] = []
    #建立一个字典来储存每个table的冷数据编号
    for embedding in cold_data_list:
        index = table_index[embedding]
        cold_dic[index].append(embedding)

    #生成冷数据
    print('start cold data')
    pbar_cold = tqdm.tqdm(total = sum(cold_data_table_counter))
    for i in range(table_num):
        table_current = cold_dic[i]
        counter = 0
        while counter < cold_data_table_counter[i]:
            access = random.choice(table_current)
            #冷数据的访问次数不能超过min_hot
            if count_list[access] + 1 > min_hot:
                continue
            count_list[access] += 1
            counter += 1
            pbar_cold.update(1)
    
    return table, count_list, embedding_num


#***********************************************************************
#@函数名称:               sample_output
#@功能描述:               将生成的数据访问记录输出到txt文件中
#@参数:                  table                列表，记录了每个table中embedding的数量
#                       embedding_num        生成数据集embedding的数量
#                       sample_num           训练集条数
#                       count_list           列表，记录每个embedding被访问的次数
#                       table_num            生成的数据集中embedding table的个数
#                       dense_data_num       每条数据中稠密数据的数量
#@返回:                  none
#***********************************************************************
def sample_output(table, embedding_num, sample_num, count_list, table_num, dense_data_num):
    #生成每个table的访问记录
    data_base = {}
    counter = 0
    for i in range(table_num):
        #print('table',i)
        data_base[i]  = []
        for embedding in range(table[i]):
            for num in range(int(count_list[counter])):
                data_base[i].append(counter)
            counter += 1
        random.shuffle(data_base[i])

    #将embedding按照热度递减进行重新编号，并生成访问记录
    tup = []
    for i in range(embedding_num):
        tup.append((i, count_list[i]))
    d = sorted(tup, key=lambda x: x[1], reverse = True)
    embedding_ID = {}
    for i in range(embedding_num):
        embedding_ID[d[i][0]] = i
    data_base_sort = {}
    for i in range(table_num):
        data_base_sort[i] = []
        for embedding in range(sample_num):
            hhah = data_base[i][embedding]
            data_base_sort[i].append(embedding_ID[hhah])
    
    #随机生成所有sample的label和稠密数据
    label_list = [str(random.randint(0,1)) for i in range(sample_num)]
    dense_data_list = [str(random.uniform(1,100)) for i in range(sample_num * dense_data_num)]
    
    #将训练数据写入txt文件
    print('start writing file1')
    f = open('data_set_1.txt','w')
    for i in tqdm.tqdm(range(sample_num)):
        output_list = [label_list[i]]
        for j in range(dense_data_num):
            output_list.append( dense_data_list[i*dense_data_num+j] )
        for k in range(table_num):
            output_list.append(str(data_base[k].pop()))
        output_list.append('\n')
        sample = " ".join(output_list)
        f.write(sample)
    f.close()
    print('finish writing file1')

    #将按热度编号的训练数据写入txt文件
    print('start writing file2')
    f = open('data_set_2.txt','w')
    for i in tqdm.tqdm(range(sample_num)):
        output_list = [label_list[i]]
        for j in range(dense_data_num):
            output_list.append( dense_data_list[i*dense_data_num+j] )
        for k in range(table_num):
            output_list.append(str(data_base_sort[k].pop()))
        output_list.append('\n')
        sample = " ".join(output_list)
        f.write(sample)
    f.close()
    print('finish writing file2')

#***********************************************************************
#@函数名称:               info_output
#@功能描述:               将生成的数据集信息输出到txt文件中
#@参数:                  table_num            生成的数据集中embedding table的个数    
#                       embedding_dimension  生成数据集中embedding的维度
#                       embedding_num        生成数据集embedding的数量
#                       sample_num           训练集条数
#                       table                每个table中embedding的数量
#@返回:                  none
#***********************************************************************
def info_output(table_num, embedding_dimension, embedding_num, sample_num, table ):
    #生成数据集的信息表
    f = open('xxx_embedding_info.txt','w')
    f.write(str(table_num) + '\n')
    f.write(str(embedding_dimension) + '\n')
    f.write(str(embedding_num) + '\n')
    f.write(str(sample_num) + '\n')
    f.write(" ".join([str(i) for i in table]) + '\n')





if __name__ == '__main__':
    #生成一百万条数据并写入txt文件中(例子)
    embedding_dimension = 512
    embedding_size = 200
    r = 0.1
    table_num = 26
    sample_num = 40000000
    dense_data_num = 5
    table, count_list, embedding_num =  embedding_access_create(embedding_dimension, embedding_size, r, table_num, sample_num)
    sample_output(table, embedding_num, sample_num, count_list, table_num, dense_data_num)
    info_output(table_num, embedding_dimension, embedding_num, sample_num, table)
    plot(count_list, embedding_num, r)
