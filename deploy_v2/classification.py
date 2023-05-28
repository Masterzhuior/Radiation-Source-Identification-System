from model import CNN
import torch
import numpy as np
import sqlite3
import datetime
# 先读取json
def load_cls_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() != 'cpu' else "cpu")
    cls_model = CNN(8192, 10).to(device)
    model_path = '/home/zwl/papercode/cnn/exp/0509_normal_scalar_sort/classfication_cnn_80_1600.pth'
    pretrain = torch.load(model_path)['model']
    cls_model.load_state_dict(pretrain)
    return cls_model

def cls_save_and_sort(data):
    data = [data]
    conn = sqlite3.connect('cls.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS data (file_name TEXT, detect_time TEXT, cls_prob TEXT, label INTEGER)')

    # 创建数据表
    cursor.executemany('INSERT INTO data VALUES (?, ?, ?, ?)', data)
    # 提交更改
    conn.commit()

    # sort by time
    cursor.execute('SELECT * FROM data ORDER BY detect_time DESC')

    table = []
    table.append(['文件名', '检测时间 ', '概率', '标签'])
    for row in cursor.fetchall():
        table.append(row)
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return table

def test_cls(file, cls_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() != 'cpu' else "cpu")
    cls_model.eval()
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    name = file['filename']
    data = np.fromfile('./data/'+name, np.int16)
    data = data[:40*8192].reshape(40, 8192)[0,...]

    data = data / np.max(np.abs(data))
    data = torch.from_numpy(data).float()
    data = torch.unsqueeze(data, dim=0)
    data = torch.unsqueeze(data, dim=0)
    with torch.no_grad():
        data = data.to(device)
        res = cls_model(data)
        res = torch.sigmoid(res)
        label = res.argmax(1).cpu().item()
        value = res.cpu().tolist()[0]
        res = {}
        for i in range(len(value)):
            res[i] = round(value[i], 2)
        res = str(res)

    return name, time, res, label

def cls_data_select_func(start_time, end_time, label):
    conn = sqlite3.connect('cls.db')
    cursor = conn.cursor()
    # 定义起始时间和终止时间
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
        # print(raw) # mak
        # print(raw)
    # 关闭数据库连接
    conn.close()
    return table

def test_file():
    pass

if __name__ == "__main__":
    data = {'filename':'data.dat'}
    cls_model = load_cls_model()
    res = test_cls(data, cls_model)
    save_and_sort(res)
