# from os.path import exists
from model import NetG
import torch
import numpy as np
import datetime
import sqlite3
import os

# 先读取json
def load_anomaly_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() != 'cpu' else "cpu")
    netg = NetG(64, 512, 1, 64, 1, 0).to(device)
    model_path = '/home/zwl/myPaperCode/1D-Ganomaly-512/output/ganomaly/phone/train/weights/netG.pth'
    pretrain = torch.load(model_path)['state_dict']
    netg.load_state_dict(pretrain)
    return netg

def test_anomaly(file, netg):
    device = torch.device("cuda:0" if torch.cuda.is_available() != 'cpu' else "cpu")
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    name = file['filename']
    data = np.fromfile('./data/'+name, np.int16)
    data = data[:40*8192].reshape(40, 8192)[0,...]

    data = data / np.max(np.abs(data))
    data = torch.from_numpy(data).float()
    data = torch.unsqueeze(data, dim=0)
    data = torch.unsqueeze(data, dim=0)

    netg.eval()
    with torch.no_grad():
        data = data.to(device)
        fake, latent_i, latent_o = netg(data)
        error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
        error = error.cpu().tolist()[0][0] 
        error = round(error, 2)
    label = 0 if error < 0.2 else 1
    return name, time, error, label


def anomaly_save_and_sort(data):
    # 将数据库中的数据按时间排序，输出结果为列表
    data = [data]
    conn = sqlite3.connect('anomaly.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS data (file_name TEXT, detect_time TEXT, abnormal_value REAL, label INTEGER)')

    # 创建数据表
    cursor.executemany('INSERT INTO data VALUES (?, ?, ?, ?)', data)
    # 提交更改
    conn.commit()

    # sort by time
    cursor.execute('SELECT * FROM data ORDER BY detect_time DESC')

    table = []
    table.append(['文件名', '检测时间 ', '异常值', '标签'])
    for row in cursor.fetchall():
        table.append(row)
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return table

def anomaly_data_select_func(start_time, end_time, label):
    conn = sqlite3.connect('anomaly.db')
    cursor = conn.cursor()
    # 定义起始时间和终止时间
    # start_time = '2023-02-24 00:00:00'
    # end_time = '2023-02-24 23:59:59'

    # 如果时间为空的话，那么取某个时间之外的所有数据 
    start_time = start_time if start_time else 0
    end_time = end_time if end_time else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 查询 detect_time 在 [start_time, end_time] 且 label=0 的所有数据
    if label:
        query = "SELECT * FROM data WHERE detect_time >= ? AND detect_time <= ? AND label=" + str(label)
    else:
        query = "SELECT * FROM data WHERE detect_time >= ? AND detect_time <= ?"
    cursor.execute(query, (start_time, end_time))
    result = cursor.fetchall()
    # print(result)
    table = []
    table.append(['文件名', '检测时间 ', '异常值', '标签'])
    for raw in result:
        table.append(list(raw))
        # print(raw)
    # 关闭数据库连接
    conn.close()
    return table


def test_file():
    # base = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/'
    file1 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/iqoo00/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-14-23-29.dat'
    file2 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/redmi28/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-10-53-07.dat'
    file3 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/iqoo02/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-14-43-52.dat'
    file4 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/iqoo06/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-14-58-40.dat'
    file5 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/redmi27/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-10-37-09.dat'
    file6 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/iqoo11/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-15-12-52.dat'
    file7 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/vivo13/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-10-24-39.dat'
    file8 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/vivo17/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-09-34-07.dat'
    file9 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/vivo22/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-10-07-03.dat'
    file10 = '/media/zwl/zwl/10ue_AllNetcom_indoor_ChinaMobileFDD_20220509/vivo25/1/460023708726918_40_B3_17250_152_123_60_10_2022-04-24-09-53-29.dat'
    # data = []
    # data.append(test_anomaly(file1, netg))
    # data.append(test_anomaly(file2, netg))
    # data.append(test_anomaly(file3, netg))
    # data.append(test_anomaly(file4, netg))
    # data.append(test_anomaly(file5, netg))
    # data.append(test_anomaly(file6, netg))
    # data.append(test_anomaly(file7, netg))
    # data.append(test_anomaly(file8, netg))
    # data.append(test_anomaly(file9, netg))
    # data.append(test_anomaly(file10, netg))

if __name__ == "__main__":
    # netg = load_anomaly_model()
    # data = './data/data.dat'
    # res = test_anomaly(data, netg)
    # print(res)
    # save_and_sort(res)
    # start_time = '2023-02-24 00:00:00'
    # start_time = None
    # end_time = '2023-02-24 23:59:59'
    # # end_time = None
    # label = 1
    data_select(start_time, end_time, label)
