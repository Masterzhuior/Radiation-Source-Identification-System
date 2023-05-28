# import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pywebio
from pywebio import *
from pywebio import config, start_server
from pywebio.input import FLOAT, input
from pywebio.output import *
from pywebio.output import put_text
from pywebio.pin import *
from anomaly import load_anomaly_model, test_anomaly, anomaly_save_and_sort, anomaly_data_select_func
from classification import load_cls_model, test_cls, cls_save_and_sort, cls_data_select_func

from new_feature import *

file = None
data = None
start_time = None
end_time = None
label = None
netg = load_anomaly_model()
cls_model = load_cls_model()

def wait_line():
    # ref: https://towardsdatascience.com/pywebio-write-interactive-web-app-in-script-way-using-python-14f50155af4e
    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)

@use_scope('scope2', clear=True)
def anomaly_data_clear():
    pass


@use_scope('scope2', clear=True)
def cls_data_clear():
    pass


def cls_select_save():
    start_time = pywebio.pin.pin['start_time']
    end_time = pywebio.pin.pin['end_time']
    label = pywebio.pin.pin['label']
    pywebio.output.close_popup()
    table = cls_data_select_func(start_time, end_time, label)
    pywebio.output.clear('scope2')
    pywebio.output.put_table(table, scope='scope2')

def anomaly_select_save():
    start_time = pywebio.pin.pin['start_time']
    end_time = pywebio.pin.pin['end_time']
    label = pywebio.pin.pin['label']
    pywebio.output.close_popup()
    table = anomaly_data_select_func(start_time, end_time, label)
    pywebio.output.clear('scope2')
    pywebio.output.put_table(table, scope='scope2')


def select_close():
    pywebio.output.close_popup()

def anomaly_data_select():
    global start_time, end_time, label
    with pywebio.output.popup('数据筛选'):
        pywebio.output.put_markdown('起始时间:')
        pywebio.pin.put_input(name='start_time')
        pywebio.output.put_markdown('终止时间:')
        pywebio.pin.put_input(name='end_time')
        pywebio.output.put_markdown('标签(0或1):')
        pywebio.pin.put_input(name='label')
        pywebio.output.put_buttons(['保存', '关闭'], onclick=[anomaly_select_save, select_close])

def cls_data_select():
    global start_time, end_time, label
    with pywebio.output.popup('数据筛选'):
        pywebio.output.put_markdown('起始时间:')
        pywebio.pin.put_input(name='start_time')
        pywebio.output.put_markdown('终止时间:')
        pywebio.pin.put_input(name='end_time')
        pywebio.output.put_markdown('标签(0-9):')
        pywebio.pin.put_input(name='label')
        pywebio.output.put_buttons(['保存', '关闭'], onclick=[cls_select_save, select_close])


def upload():
    global file
    file=pywebio.input.file_upload("上传文件...")
    os.makedirs('./data', exist_ok=True)
    # 将用户上传的数据保存在data目录下
    if file == None:
        show_msg('不能上传空数据!')
    open('./data/'+file['filename'], 'wb').write(file['content'])
    check_data()

def check_data():
    try:
        data = np.fromfile('./data/'+file['filename'], np.int16)
        L = len(data) // 8192
        data = data[:L*8192].reshape(40, 8192)[0, :]
    except:
        pywebio.output.toast('文件异常, 请重新上传文件', position='center', color='error', duration=3)

@ use_scope('scope1', clear= True)
def visualize():
    import plotly.express as px
    data = np.fromfile('./data/'+file['filename'], np.int16)
    L = len(data) // 8192
    # chose 1 sambol
    data = data[:L*8192].reshape(L, 8192)[0, :]
    data = data / np.max(np.abs(data))
    fig1=px.line(data)
    fig1_html=fig1.to_html(include_plotlyjs= 'require')
    # with use_scope('scope2', clear=True):  # 创建scope1, 并先清除内容
    #     pywebio.output.put_html(fig1_html, scope='scope1')

    # fft
    fft=np.fft.fft(data)
    fft=np.abs(fft)[5192: 8192]
    fig2=px.line(fft)
    fig2_html=fig2.to_html(include_plotlyjs= 'require')
    pywebio.output.put_html('<br>')
    pywebio.output.put_collapse('数据可视化结果', [
        pywebio.output.put_html(fig1_html, scope='scope1'),
        pywebio.output.put_html(fig2_html, scope='scope2'),
    ], open= True)


@ use_scope('scope1', clear= True)
def anomaly_detection():
    '''
    异常检测部分输出表格： 数据名称 检测时间 异常值 label
    '''
    res = test_anomaly(file, netg)
    table = anomaly_save_and_sort(res)
    
    pywebio.output.put_html('<br>')
    # pywebio.output.put_scrollable(pywebio.output.ues_scope('scope1'), height=200, keep_bottom=False)
    pywebio.output.put_collapse('数据异常检测结果',
                                [
                                    pywebio.output.put_buttons(['数据筛选', '数据清除'], onclick=[
                                                               anomaly_data_select, anomaly_data_clear], outline=True),
                                    pywebio.output.put_scope('scope2', content=[
                                        pywebio.output.put_table(table),
                                    ]),
                                ], open = True)


@ use_scope('scope1', clear = True)
def classification():
    res = test_cls(file, cls_model)
    table = cls_save_and_sort(res)
    pywebio.output.put_html('<br>')
    pywebio.output.put_collapse('数据分类结果',
                                [
                                    pywebio.output.put_buttons(['数据筛选', '数据清除'], onclick=[
                                                               cls_data_select, cls_data_clear], outline=True),
                                    pywebio.output.put_scope('scope2', content=[
                                        pywebio.output.put_table(table),
                                    ]),
                                ], open = True
                                )


def ui():
    img=open('./fig/phone.png', 'rb').read()
    pywebio.output.put_row(
        [
            pywebio.output.put_image(img, width='100px', height='100px'),
            pywebio.output.put_text('手机辐射源识别项目系统').style(
                'color:black; font-size: 30px; height: 100px; line-height: 100px; text-align: center; padding-right: 120px; padding-top: 10px')
        ],
        size='20% 80%'  # 调整这一row的元素位置比例
    )
    pywebio.output.put_markdown('---')

    img = open('./fig/upload.png', 'rb').read()
    pywebio.output.put_row(
        [
            pywebio.output.put_button(
                ['上传检测文件'], onclick=upload, outline=True).style('text-align: left'),
            pywebio.output.put_button(
                ['更新训练数据'], onclick=update_training_data, outline=True).style('text-align: center'),
            pywebio.output.put_button(
                ['重新训练模型'], onclick=retrain_model, outline=True).style('text-align: right'),
        ],
    )
    pywebio.output.put_markdown('---')

    pywebio.output.put_text('功能选择').style(
        'text-align: center; font-size: 25px; margin:0em 0em 1em 0em')  # 段间距
    pywebio.output.put_row(
        [
            pywebio.output.put_buttons(
                ['数据可视化'], onclick=[visualize], outline=True).style('text-align: left'),
            pywebio.output.put_buttons(
                ['数据异常检测'], onclick=[anomaly_detection], outline=True).style('text-align: center'),
            pywebio.output.put_buttons(
                ['数据分类'], onclick=[classification], outline=True).style('text-align: right'),
        ],
    )

@config(theme='sketchy')
def main():
    ui()

if __name__ == "__main__":
    start_server(main, port=36535, debug=True)
