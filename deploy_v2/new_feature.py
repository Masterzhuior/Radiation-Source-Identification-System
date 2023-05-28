from pywebio.input import actions, select
from pywebio import input
import os
import sqlite3
import subprocess


def update_training_data():
    # 应该有两个功能
    # 1. 数据库中数据进行筛选: 然后将这些类别数据添加到每个数据库中
    # 2. 主动上传数据: 选择数据，然后上传到某个类别中

    res = actions('选择您想执行的功能', [{'label': '数据筛选','value':'filter','color': 'light'}, {'label' : '数据上传','value':'upload','color':'light'}], help_text='数据筛选将从数据库中对数据进行筛选，然后将数据添加到训练数据库中；数据上传则为直接上传到某个文件夹内')
    if res == 'filter':
        filter = actions('选择您想执行的功能', [{'label': '筛选异常数据','value':'filter_anomaly','color': 'light'}, {'label' : '筛选分类数据','value':'filter_cls','color':'light'}], help_text='')
        if filter == 'filter_anomaly':
            # 筛选异常数据
            #连接到数据库
            conn = sqlite3.connect('anomaly.db')
            #获取游标对象
            cursor = conn.cursor()
            #打印出所有列名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            tablename = tables[0][0]
            #执行SQL查询语句，获取所有列名, data是table name
            cursor.execute(f"PRAGMA table_info('{tablename}')")
            # columns = [col[1] for col in cursor.fetchall()]
            cursor.execute(f"SELECT file_name FROM {tablename} WHERE label=1")
            file_names = cursor.fetchall()
            # 获取完文件名后，创建新类别，然后将异常文件移动到新文件夹里面
            file_names = list(set(file_names)) # 去重
            new_cls = len(os.listdir('mini_dataset/'))
            new_path = './mini_dataset/' + str(new_cls) + '/' 
            os.mkdir(new_path) # 假设只有一个新类别

            # 移动数据
            for file_name in file_names:
                for file in os.listdir('data'):
                    if file == file_name[0]:
                        os.rename('data/'+file, new_path + file)

            print('success')
            #关闭数据库连接
            cursor.close()
            conn.close()
        else:
            # 筛选分类数据
            conn = sqlite3.connect('cls.db')
            #获取游标对象
            cursor = conn.cursor()
            #打印出所有列名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            tablename = tables[0][0]
            #执行SQL查询语句，获取所有列名, data是table name
            cursor.execute(f"PRAGMA table_info('{tablename}')")
            columns = [col[1] for col in cursor.fetchall()]
            # 执行查询语句
            cursor.execute(f"SELECT MAX(label) FROM {tablename}")
            # 获取类别最大值
            result = cursor.fetchone()
            for i in range(result+1):
                cursor.execute(f"SELECT file_name FROM {tablename} WHERE label={i}")
                file_names = cursor.fetchall()
                # 获取完文件名后，创建新类别，然后将异常文件移动到新文件夹里面
                file_names = list(set(file_names)) # 去重
                path = './mini_dataset/' + str(i) + '/' 
                # 移动数据
                for file_name in file_names:
                    for file in os.listdir('data'):
                        if file == file_name[0]:
                            os.rename('data/'+file, path + file)
            print('success')
            #关闭数据库连接
            cursor.close()
            conn.close()

    elif res == 'upload':
        # 检测所有类别
        data_path = './mini_dataset/'
        cls = sorted(os.listdir(data_path))
        # 显示到可选的类别
        chose_cls = select(label='选择类别文件夹', options=cls)
        file = input.file_upload("上传文件...")
        # 添加文件
        open('./mini_dataset/'+chose_cls+file['filename'], 'wb').write(file['content'])


def retrain_model():
    # 1. 重新训练分类模型
    # 2. 重新训练异常检测模型
    res = actions('选择您想执行的功能', [{'label': '重新训练分类模型','value':'retrain_cls_model','color': 'light'}, {'label' : '重新训练异常检测模型','value':'retrain_anomoly_model','color':'light'}], help_text='')
    # 检测类别数量，因为前面添加了类别，数量可能会变化
    # data_path = './mini_dataset/'
    # num_cls = len(os.listdir('mini_dataset/'))

    # 重新训练分类模型
    if res == 'retrain_cls_model':
        cmd = ['sh', '/home/zwl/papercode/cnn/exp_no_process_sort.sh']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        for line in iter(p.stdout.readline, b''):
            print(line.decode(), end='')

    # 重新训练异常检测模型
    else:
        cmd = ['sh', '/home/zwl/papercode/1D-Ganomaly-512/experiments/run_phone.sh']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        for line in iter(p.stdout.readline, b''):
            print(line.decode(), end='')
