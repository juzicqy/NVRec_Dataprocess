#-----------------------------------------------------------------------------------
#         处理按照热度排序的数据集，将embedding按照table type排序并生成新数据集
#-----------------------------------------------------------------------------------


with open ('Avazu_dataset_reid.txt','r') as f:
    samples = f.readlines()

ID_list = [0] * 9449205
table = [7, 7, 4737, 7745, 26, 8552, 559, 36, 2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]



dic = {}
for i in range(21):
    dic[i] = []
for sample in samples:
    list_sample = sample.split(" ")
    embeddings = list_sample[-22:-1]
    for i in range(len(table)):
        dic[i].append(int(embeddings[i]))

print('finish reading')
dic_count = {}
for i in range(21):
    dic_count[i] = []
    hh = {}
    for access in dic[i]:
        if access not in hh:
            hh[access] = 1
        else:
            hh[access] += 1
    for key in hh:
        dic_count[i].append(key)
    dic_count[i].sort()
print('start re_ID')

table_tup = []
for i in range(21):
    table_tup.append((i, table[i]))
d = sorted(table_tup, key=lambda x: x[1])
print(d)

id = 0
for i in range(21):
    for embedding in dic_count[d[i][0]]:
        ID_list[ embedding ] = id
        id += 1

print('start writing')
f = open('Avazu_dataset_type.txt','w')
for sample in samples:
    list_sample = sample.split(" ")
    embeddings = list_sample[-22:-1]
    output_list = list_sample[0:-22]
    for i in range(21):
        access = int(embeddings[i])
        output_list.append(str(ID_list[access]))
    output_list.append('\n')
    sample = " ".join(output_list)
    f.write(sample)
f.close()




    
    





