import multiprocessing
import time
import tqdm

#-----------------------------------------------------------------------------------
#                     尝试采用多进程的方法以加快扫描数据集的速度
#-----------------------------------------------------------------------------------

with open ('synthetic_dataset_reid.txt','r') as f:
    samples = f.readlines()

#将数据分成多份用于多进程遍历数据集
samples_slice = []
for i in range(23):
    samples_slice.append( samples[i*1666666: (i+1)*1666666] )
samples_slice.append(samples[23*1666666: ])

#遍历其中一份数据集
def data_scanning(n):
    print(n)
    count_dic = {}
    samples_now = samples_slice[n]
    
    for sample in samples_now:
        list_sample = sample.split(" ")
        embeddings = list_sample[-27:-1]
        for access in embeddings:
            if access not in count_dic:
                count_dic[access] = 1
            else:
                count_dic[access] += 1
    return count_dic
    
#利用multiprocessing pool多进程遍历数据集
if __name__ == "__main__":
    
    pool = multiprocessing.Pool(processes=24)
    print('start')
    time0 = time.time()
    arg_list = list(range(24))
    pool.map( data_scanning, arg_list )
    pool.close()
    pool.join()
    time1 = time.time()
    print(time1-time0)


"""
单进程遍历数据集
if __name__ == "__main__":
    count_dic = {}
    print('start')
    time1 = time.time()
    for sample in samples:
        list_sample = sample.split(" ")
        embeddings = list_sample[-27:-1]
        for access in embeddings:
            if access not in count_dic:
                count_dic[access] = 1
            else:
                count_dic[access] += 1
    time2 = time.time()
    print(time2-time1)
"""