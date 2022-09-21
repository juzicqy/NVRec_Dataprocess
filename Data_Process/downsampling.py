import random
import tqdm


#-----------------------------------------------------------------------------------
#                    随机采样部分数据集，统计访问频率并编号，生成新数据集文件
#-----------------------------------------------------------------------------------


with open ('kaggle_dataset_reid.txt','r') as f:
    samples = f.readlines()

#随机挑选部分数据集
#sample_5 = random.sample( samples, int(len(samples) * 0.05) )
#sample_10 = random.sample( samples, int(len(samples) * 0.1) )
sample_20 = random.sample( samples, int(len(samples) * 0.2) )

#遍历下采样数据集，记录访问次数
count_dic = {}
for i in range(26):
    count_dic[i] = {}
for sample in tqdm.tqdm(sample_20):
    list_sample = sample.split(" ")
    embeddings = list_sample[-27:-1]
    for i in range(26):
        if int(embeddings[i]) not in count_dic[i]:
            count_dic[i][int(embeddings[i])] = 1
        else:
            count_dic[i][int(embeddings[i])] += 1

#重新编号下采样数据集中的数据
tup_list = []
for i in range(26):
    for embedding in count_dic[i]:
        tup_list.append( (embedding, count_dic[i][embedding]) )
a = sorted(tup_list, key=lambda x: x[1], reverse=True)


ID_dic = {}
for i in range(len(tup_list)):
    ID_dic[a[i][0]] = i

#输出使用新编号的下采样数据集
f = open('kaggle_dataset_20.txt','w')
for sample in tqdm.tqdm(sample_20):
    list_sample = sample.split(" ")
    embeddings = list_sample[-27:-1]
    output_list = list_sample[0:-27] + [str(ID_dic[int(embeddings[i])]) for i in range(26)] + ['\n']
    sample = " ".join(output_list)
    f.write(sample)
f.close()
