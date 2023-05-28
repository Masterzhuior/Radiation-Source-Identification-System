import sqlite3

conn = sqlite3.connect('data.db')

cursor = conn.cursor()

# 创建数据表
cursor.execute('CREATE TABLE IF NOT EXISTS data (file_name TEXT, detect_time TEXT, abnormal_value REAL, label INTEGER)')

# 插入数据
data = [('a.txt', '2022-01-01', 0.8, 1), ('b.txt', '2022-02-01', 0.3, 0), ('c.txt', '2022-03-01', 0.6, 1)]

cursor.executemany('INSERT INTO data VALUES (?, ?, ?, ?)', data)

# 提交更改
conn.commit()

# 关闭游标和连接
cursor.close()
conn.close()

