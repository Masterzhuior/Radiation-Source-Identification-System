# Radiation-Source-Identification-System
## 主要使用的模块：

+ PyWebIO
+ pytorch

PyWebIO能够让我们使用Python来进行前端的编写，通过Python代码，和css可以控制内部的元素样式。

## 代码目录结构：
+ anomaly.db        # 异常检测数据库
+ anomaly.py        # 异常检测代码
+ app.py            # 程序代码入口
+ classification.py # 分类文件
+ cls.db            # 分类数据库
+ data              # 上传数据的存放地点
+ fig               # 界面的logo图片
+ mini_dataset      # 测试的迷你数据集
+ model.py          # 使用到的深度学习模型
+ new_feature.py    # 新功能文件
+ test              # 测试文件

## Run the code
```
python3 app.py
```

## 函数介绍:

### app.py
该文件中主要有ui函数, 为整个UI界面的启动，包括所有的功能，都写在该函数中。最早实现的功能有：

+ 数据上传
+ 数据可视化
+ 数据分类: 对上传的数据进行分类，点击数据分类模块后，就会对数据分类，分类后，可以清除数据结果，以及数据筛选
+ 异常检测: 对上传的数据进行异常检测，点击数据异常检测模块后，就会调用异常检测函数，同样可以清除数据结果，以及数据筛选
+ 
后面补充两个新的功能：

+ 重新训练模型: 重新训练异常检测模型和分类模型
+ 更新训练数据: 将检测的数据，添加到训练数据中，和更新训练模型功能配合

### start_server

启动UI界面，通过端口localhost:36536可以进行访问

在PyWebIO中，start_server()函数用于启动一个Web服务器，并将带有PyWebIO应用程序的页面服务。debug参数是可选的，用于设置调试模式。

当debug设置为True时，PyWebIO将在运行期间输出详细的调试信息，在开发和调试应用程序时非常有用。此时，运行PyWebIO应用程序时，Web浏览器将现在在调试模式下，同时Python控制台将显示详细的应用程序运行日志。

但是，当准备将应用程序部署到生产环境时，应将debug设置为False或省略该参数。这样可以避免在生产环境中泄露敏感信息，并提高应用程序的安全性。

### ui
查看pywebio.output.put_row函数，里面有很多put_buttons功能。

每个button都是界面上的一个功能按键，点击对应的按键，就能调用其中的函数。调用onclick后面的函数

