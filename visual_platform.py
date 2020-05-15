# import necessary package
"""
固定格式可视化工具
版本：matplotlib   3.2.1
      numpy        1.18.0
      pandas       0.25.3
"""

import numpy as np
import math
import random
import os
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close()
# 主要色系：设置以下颜色为循环调用


color_dict = {
    # col1：明亮风
    1: ['#0e72cc', '#6ca30f', '#f59311', '#fa4343', '#16afcc', '#85c021', '#d12a6a', '#0e72cc', '#6ca30f', '#f59311',
        '#fa4343', '#16afcc'],
    # col2: 经典风
    2: ['#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac'],
    # col3: 清冷风
    3: ['#45c8dc', '#854cff', '#5f45ff', '#47aee3', '#d5d6d8', '#96d7f9', '#f9e264', '#f47a75', '#009db2', '#024b51',
        '#0780cf', '#765005'],
    # col4: 暗沉风
    4: ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#264478', '#9E480E', '#636363', '#997300',
        '#255E91', '#43682B'],
    # col5: 渐变风
    5: ['#ED7D31', '#FFC000', '#70AD47', '#9E480E', '#997300', '#43682B', '#F1975A', '#FFCD33', '#8CC168', '#D26012',
        '#CC9A00', '#5A8A39'],
    # col6: 雅致风
    6: ['#FDB462', '#80B1D3', '#FB8072', '#D06F03', '#346F97', '#D51B06', '#FDC381', '#99C1DC', '#FC998E', '#FC931D',
        '#4E92C2', '#F9402B'],
}


class LineChart(object):
    # 静态变量 用于子图时子生成单个类
    __first_init = True
    __species = None

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, data, figsize=(8, 5), dpi=300, subplots_shape=None):
        self.data = data.copy()
        self.figsize = figsize
        self.dpi = dpi
        if subplots_shape is None and self.__first_init:
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, dpi=self.dpi)
            LineChart.__first_init = True
            print('sing')
        elif self.__first_init:
            self.fig, self.ax = plt.subplots(
                subplots_shape[0], subplots_shape[1], figsize=self.figsize, dpi=self.dpi)
            LineChart.__first_init = False

    def draw(self,
             title,
             x_label,
             y_label,
             title_size=20,
             x_font_size=16,
             y_font_size=16,
             style=1,
             show_x_grid=False,
             show_y_grid=True,
             ax=None,
             save_fig=True):

        if style == 1:  # 报告格式
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False
            for i in range(self.data.shape[1] - 1):
                ax.plot(self.data.iloc[:, 0], self.data.iloc[:, i + 1], '-o')
                # 设置数据标签
                for x, y in zip(
                        self.data.iloc[:, 0], self.data.iloc[:, i + 1]):
                    ax.text(x, y + 0.3, str(y), ha='center', va='bottom')
            ax.set_title('%s' % title, size=title_size)
            ax.set_xlabel(x_label, size=x_font_size)  # 设置X轴名称及字体大小
            ax.set_ylabel(y_label, size=y_font_size)  # 设置X轴名称及字体大小
            if show_y_grid:
                ax.grid(
                    True,
                    axis='y',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            if show_x_grid:
                ax.grid(
                    True,
                    axis='x',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延x轴的)
            ax.legend(loc='upper right')  # 设置标签位置
            # 去掉边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if save_fig:
                plt.savefig('t1.jpg')

        elif style == 2:  # PPT格式
            show_x_grid = True
            show_y_grid = True
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False

            for i in range(self.data.shape[1] - 1):
                ax.plot(self.data.iloc[:, 0], self.data.iloc[:, i + 1], '-o')
                # 设置数据标签
                for x, y in zip(
                        self.data.iloc[:, 0], self.data.iloc[:, i + 1]):
                    ax.text(x, y + 0.3, str(y), ha='center', va='bottom')

            ax.set_title('%s' % title, size=title_size)
            ax.set_xlabel(x_label, size=x_font_size)  # 设置X轴名称及字体大小
            ax.set_ylabel(y_label, size=y_font_size)  # 设置X轴名称及字体大小
            if show_y_grid:
                ax.grid(
                    True,
                    axis='y',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            if show_x_grid:
                ax.grid(
                    True,
                    axis='x',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            ax.legend(loc='best')  # 设置标签位置
            if save_fig:
                plt.savefig('t1.jpg')

        elif style == 3:
            show_x_grid = True
            show_y_grid = True

            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False

            self.ax.patch.set_facecolor("#D3D3D3")  # 设置ax1区域背景颜色

            for i in range(self.data.shape[1] - 1):
                plt.plot(self.data.iloc[:, 0], self.data.iloc[:, i + 1], '-o')
                # 设置数据标签
                for x, y in zip(
                        self.data.iloc[:, 0], self.data.iloc[:, i + 1]):
                    plt.text(x, y + 0.3, str(y), ha='center', va='bottom')

            plt.title('%s' % title, size=title_size)
            plt.xlabel(x_label, size=x_font_size)  # 设置X轴名称及字体大小
            plt.ylabel(y_label, size=y_font_size)  # 设置X轴名称及字体大小

            if show_y_grid:
                self.ax.grid(
                    True,
                    axis='y',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            if show_x_grid:
                self.ax.grid(
                    True,
                    axis='x',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            self.ax.legend(loc='best')  # 设置标签位置

        elif style == 4:
            show_x_grid = False
            show_y_grid = False
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False

            for i in range(self.data.shape[1] - 1):
                plt.plot(self.data.iloc[:, 0], self.data.iloc[:, i + 1], '-o')
                # 设置数据标签
                for x, y in zip(
                        self.data.iloc[:, 0], self.data.iloc[:, i + 1]):
                    plt.text(x, y + 0.3, str(y), ha='center', va='bottom')

            plt.title('%s' % title, size=title_size)
            plt.xlabel(x_label, size=x_font_size)  # 设置X轴名称及字体大小
            plt.ylabel(y_label, size=y_font_size)  # 设置X轴名称及字体大小
            if show_y_grid:
                self.ax.grid(
                    True,
                    axis='y',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            if show_x_grid:
                self.ax.grid(
                    True,
                    axis='x',
                    alpha=0.8,
                    linestyle='--')  # 添加网格线(只要延y轴的)
            self.ax.legend(loc='best')  # 设置标签位置


class PieChart(object):
    # 静态变量 用于子图时子生成单个类
    __first_init = True
    __species = None

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, data, figsize=(8, 5), dpi=300, subplots_shape=None):
        self.data = data.copy()
        self.figsize = figsize
        self.dpi = dpi
        if subplots_shape is None and self.__first_init:
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, dpi=self.dpi)
            PieChart.__first_init = True
            print('sing')
        elif self.__first_init:
            self.fig, self.ax = plt.subplots(
                subplots_shape[0], subplots_shape[1], figsize=self.figsize, dpi=self.dpi)
            PieChart.__first_init = False

    def draw(self,
             title,
             title_size=20,
             explode=None,  # 分裂值
             show_legend=True,  # 显示图例
             style=1,
             ax=None,
             save_fig=True):

        if style == 1:
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False

            # pd数据转为list
            labels = self.data['label'].tolist()
            values = self.data['V1'].tolist()
            # 分裂饼图
            ax.pie(values,
                   # explode分裂属性：用于设置各个部分分裂程度
                   explode=explode,
                   # 设置label属性
                   labels=labels,
                   # autopct属性：设置百分比位数
                   autopct='%3.1f%%',
                   # startangle属性：设置起始角度
                   startangle=180,
                   # shadow阴影属性设置
                   shadow=False,
                   # colors各扇形部分颜色设置
                   # colors = ['#3CB371', '#FFD700', 'gray', '#1E90FF',
                   # '#FFA500'])
                   colors=['c', 'r', 'y', 'g', 'gray'])
            # 标题
            ax.set_title(title, size=title_size)
            # axis的equal属性用于调节饼图的形状，不至于因画布大小的设置（非1:1）而变形，保证饼图是一个正圆
            ax.axis('equal')
            # legend属性用于调节图例的位置，bbox_to_anchor用于调整位置
            if show_legend:
                ax.legend(bbox_to_anchor=(0.95, 0.8))
            if save_fig:
                plt.savefig('t1.jpg')

        elif style == 2:
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False
            # pd数据转为list
            labels = self.data['label'].tolist()
            values = self.data['V1'].tolist()
            c1 = (112, 173, 71)
            c2 = (255, 217, 72)
            colors = [c1, c2]
            col = [[i / 255. for i in c] for c in colors]
            width = 0.35
            text = "达标城市情况"
            patches, texts, autotexts = ax.pie(
                values, radius=0.8, autopct='%3.1f%%', pctdistance=1 - width / 2, labels=labels, colors=col,
                startangle=180)
            plt.setp(patches, width=width, edgecolor='white')
            ax.text(
                0,
                0,
                text,
                ha='center',
                size=16,
                fontweight='bold',
                va='center')  # 设置环形图中间文本的属性
            # 标题
            ax.set_title(title, size=title_size)
            if save_fig:
                plt.savefig('t1.jpg')

        elif style == 3:
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False
            # pd数据转为list
            labels = self.data['label'].tolist()
            values_1 = self.data['V1'].tolist()
            values_2 = self.data['V2'].tolist()
            colors = ['c', 'r', 'y', 'g', 'gray']
            # 外环
            patches_1, texts_1, autotexts_1 = plt.pie(values_1,
                                                      autopct='%3.1f%%',
                                                      radius=1,
                                                      pctdistance=0.85,  # 饼图重心到autotexts对象的相对距离
                                                      colors=colors,
                                                      labels=labels,
                                                      startangle=180,
                                                      textprops={'color': 'w'},
                                                      wedgeprops={
                                                          'width': 0.3, 'edgecolor': 'w'}
                                                      )

            # 内环
            patches_2, texts_2, autotexts_2 = plt.pie(values_2,
                                                      autopct='%3.1f%%',
                                                      radius=0.7,
                                                      pctdistance=0.75,  # 饼图重心到autotexts对象的相对距离
                                                      colors=colors,
                                                      startangle=180,
                                                      # 饼图中百分比文本的属性字典
                                                      textprops={'color': 'w'},
                                                      wedgeprops={
                                                          'width': 0.3, 'edgecolor': 'w'}
                                                      # 饼图的格式，width设置了环的宽度，edgecolor设置了边缘颜色
                                                      )
            # 图例
            ax.legend(patches_1,
                      labels,  # 图例列表
                      fontsize=8,  # 图例大小
                      title='公司列表',  # 图例名称
                      loc='center right',  # 图例居右
                      bbox_to_anchor=(1.2, 0.6)  # 调节图例位置
                      )

            # 设置文本样式
            plt.setp(autotexts_1, size=8, weight='bold')
            plt.setp(autotexts_2, size=8, weight='bold')
            plt.setp(texts_1, size=8)
            text = "达标城市情况"
            ax.text(
                0,
                0,
                text,
                ha='center',
                size=16,
                fontweight='bold',
                va='center')  # 设置环形图中间文本的属性
            ax.axis('equal')
            ax.set_title(title, size=title_size)
            if save_fig:
                plt.savefig('t1.jpg')


class BarChart(object):
    # 静态变量 用于子图时子生成单个类
    __first_init = True
    __species = None

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, data, figsize=(8, 5), dpi=300, subplots_shape=None):
        self.data = data.copy()
        self.figsize = figsize
        self.dpi = dpi
        if subplots_shape is None and self.__first_init:
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, dpi=self.dpi)
            BarChart.__first_init = True
            print('single')
        elif self.__first_init:
            self.fig, self.ax = plt.subplots(
                subplots_shape[0], subplots_shape[1], figsize=self.figsize, dpi=self.dpi)
            BarChart.__first_init = False

    def draw(self,
             ax=None,
             title='',
             xlabel='',
             ylabel='',
             style=1,
             show_grid_type=None,
             show_spines_list=None,
             bar_width=None,
             top_height=0.4,
             fontsize=None,
             title_size=None,
             show_legend=True,
             ylim=None,
             save_fig=True,
             show_gray_background=None,
             show_bar_edge=False,
             decimal_length=1,
             color=None,
             ):
        """
        :param ax: 绘制图的axis
        :param title: 标题
        :param xlabel: x轴标签
        :param ylabel: y轴标签
        :param style: 绘图风格， 类型整数[1,2,3,4,5.。。]
        :param show_grid_type:是否显示网格
        :param show_spines_list:显示上下左右那些边框，按[上，下，左 ，右]顺序,True为显示False不显示
        :param bar_width: 柱子宽度
        :param top_height:柱子上标注离柱顶高度
        :param fontsize:柱子标注字体大小
        :param title_size:标题字体大小
        :param show_legend:是否显示图例
        :param ylim: y刻度最大值
        :param save_fig:是否保存图像
        :param show_gray_background:  是否显示灰色背景
        :param show_bar_edge: 是否显示柱子黑色边框
        :param decimal_length: 柱子标注小数位保留位数
        :param color:  选择配色风格
        :return: 无返回 ，使用全局plt显示绘图
        """
        style_dict = {
            1: {"color": color_dict[6],
                "show_gray_background": False,
                "show_grid": "both",
                "show_spines_list": [False, True, False, True], },
            2: {"color": color_dict[3],
                "show_gray_background": True,
                "show_grid": "both",
                "show_spines_list": [False, False, False, False]},
            3: {"color": color_dict[6],
                "show_gray_background": False,
                "show_grid": False,
                "show_spines_list": [False, False, False, False],

                },
        }
        # 调整字体编码
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        # 若不传入ax则使用自身ax,仅在单子图下可以不穿，多子图必须传入
        if ax is None:
            ax = self.ax
        # pandas数据转为list
        labels = self.data.values.T.tolist()[0]
        values = self.data.values.T.tolist()[1:]

        # 判断是否指定参数
        if bar_width == None:
            bar_width = round(1.0 / (len(values) + 1), 2)  # 根据数据维度自动设置柱子宽度
        if title_size is None:
            title_size = 20
        if fontsize is None:
            fontsize = 6
        if show_gray_background is None:
            show_gray_background = style_dict[style]["show_gray_background"]
        if show_grid_type is None:
            show_grid_type = style_dict[style]["show_grid"]
        if show_spines_list is None:
            show_spines_list = style_dict[style]["show_spines_list"]

        if color is None:
            color = style_dict[style]["color"]
            print(style, color)
        if ylim is None:
            plt.ylim([0, math.ceil(max(max(values)) + 1)])  # 设置y轴刻度范围，若不指定设置为数据最大值+1
        else:
            plt.ylim([0, ylim])  # 按指定最大y轴赋值

        assert show_grid_type in [None, False, 'x', 'y', 'both'], 'grid type set wrong'

        for i in range(len(values)):
            if style == 1:
                ax.bar(np.array(range(len(values[i]))) + bar_width * i,
                       values[i],  # 数据
                       fc=color[i],  # 调用色系
                       width=bar_width,  # 设置柱子宽度
                       tick_label=labels,  # 设置横坐标名称

                       )
            elif style == 2:
                ax.bar(np.array(range(len(values[i]))) + bar_width * i,
                       values[i],  # 数据
                       fc=color[i],  # 调用色系
                       width=bar_width,  # 设置柱子宽度
                       tick_label=labels,  # 设置横坐标名称
                       )

            elif style == 3:
                ax.bar(np.array(range(len(values[i]))) + bar_width * i,
                       values[i],  # 数据
                       fc=color[i],  # 调用色系
                       width=bar_width,  # 设置柱子宽度
                       tick_label=labels,  # 设置横坐标名称
                       edgecolor= 'black',  # edgecolor属性用于设置每个柱边框的颜色，不可调
                        linewidth= 0.7,  # linewidth属性用于设置每个柱边框线条粗细程度,不可调
                       )

            # 显示标签，设置精确度、fontsize可调
            for a, b in zip(np.arange(len(values[0])), values[i]):
                ax.text(a + bar_width * i,  # 设置数值标签的横坐标位置
                        b + top_height,  # 设置数值标签的纵坐标位置，距离柱顶部的位置，可调
                        round(b, decimal_length),  # 设置数值标签的精确度，可调
                        ha='center',  # 水平对齐方式，居中，不可调
                        va='top',  # 垂直对齐方式，居上，不可调
                        fontsize=fontsize)  # 标签字体大小，可调

            # 添加网格线(axis参数x,y,both,分别表示只显示x轴线或y轴线或全部显示)
            if show_grid_type:
                ax.grid(True, axis=show_grid_type, alpha=0.5, linestyle='--')  # 其中,axis参数可调，有x，y，both三种形式
            if show_gray_background:
                ax.patch.set_facecolor('#EAEAF2')  # 设置背景颜色为灰色

            ## 是否显示边框（若：不显示，则对应的变为False）
            for i in show_spines_list:
                ax.spines['top'].set_visible(i)  ## 上边框，可调，Fasle为不显示
                ax.spines['bottom'].set_visible(i)  ## 下边框，可调，True为显示
                ax.spines['left'].set_visible(i)  ## 左边框，可调，True为显示
                ax.spines['right'].set_visible(i)  ## 右边框，可调，Fasle为不显示
            ##设置柱状图在网格线上面
            ax.set_axisbelow(True)
            # 标题 x 标签  y标签
            ax.set_title(title, size=title_size)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if show_legend:
                plt.legend(self.data.columns.tolist()[1:], bbox_to_anchor=(0.99, 0.95), prop={'size': 7})
            if save_fig:
                plt.savefig('out.jpg')


class BoxChart(object):
    # 静态变量 用于子图时子生成单个类
    __first_init = True
    __species = None

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, data, figsize=(8, 5), dpi=300, subplots_shape=None):
        self.data = data.copy()
        self.figsize = figsize
        self.dpi = dpi
        if subplots_shape is None and self.__first_init:
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, dpi=self.dpi)
            BoxChart.__first_init = True
            print('sing')
        elif self.__first_init:
            self.fig, self.ax = plt.subplots(
                subplots_shape[0], subplots_shape[1], figsize=self.figsize, dpi=self.dpi)
            BoxChart.__first_init = False

    def draw(self,
             title,
             title_size=20,
             style=1,
             colors=['pink', 'lightblue', 'lightgreen'],
             ax=None,
             save_fig=True,
             showmeans=True,
             meanline=False):

        if style == 1:
            # 调整字体编码
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False

            # pd数据转为list
            # labels = self.data['label'].tolist()
            values_1 = self.data['V1'].tolist()
            values_2 = self.data['V2'].tolist()
            values_3 = self.data['V3'].tolist()
            all_data = [values_1, values_2, values_3]
            bplot = ax.boxplot(
                all_data,
                vert=True,
                patch_artist=True,
                showmeans=showmeans,
                meanline=meanline)  # meanline=False，那么均值位置会在图中用小三角表示出来
            if colors is not None:
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
            # axes[0]表示在第一张图的轴上描点画图
            # vert=True表示boxplot图是竖着放的
            # patch_artist=True 表示填充颜色
            ax.grid(True, axis='both', alpha=0.5, linestyle='--')
            if save_fig:
                plt.savefig('t1.jpg')


class HeatmapChart(object):
    # 静态变量 用于子图时子生成单个类
    __first_init = True
    __species = None

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, data, figsize=(8, 5), dpi=300, subplots_shape=None):
        self.data = data.copy()
        self.figsize = figsize
        self.dpi = dpi
        if subplots_shape is None and self.__first_init:
            self.fig, self.ax = plt.subplots(
                figsize=self.figsize, dpi=self.dpi)
            HeatmapChart.__first_init = True
            print('sing')
        elif self.__first_init:
            self.fig, self.ax = plt.subplots(
                subplots_shape[0], subplots_shape[1], figsize=self.figsize, dpi=self.dpi)
            HeatmapChart.__first_init = False

    def draw(self,
             title,
             title_size=20,
             style=1,
             ax=None,
             save_fig=True):
        pass
        # 暂时未使用
        # if style == 1:
        #     # 调整字体编码
        #     mpl.rcParams['font.sans-serif'] = ['SimHei']
        #     mpl.rcParams['axes.unicode_minus'] = False
        #
        #     target = pd.pivot_table(self.data, index='month')
        #
        #     f, ax = plt.subplots(figsize=(10, 6), dpi=100)
        #     ax.set_title(title, fontsize=18)
        #     sns.heatmap(target, annot=True, fmt='.0f', ax=ax)
        #     if save_fig:
        #         plt.savefig('t1.jpg')


if __name__ == '__main__':
    # 折线图单个图与子图绘制示例
    data1 = pd.DataFrame({'label': ['Feb', 'Jan', 'Mar', 'Apr', 'May', 'Aug', 'Seb'],
                          'V1': [5, 8, 4.3, 7, 8.2, 4.6, 8],
                          'V2': [6, 7, 8.2, 6, 4.5, 7, 8.6],
                          'V3': [7, 8, 7.2, 4, 3.5, 8, 8.9],
                          'V4': [5, 5, 8.2, 6, 4.5, 6, 9.6],
                          # 'V5': [6, 7, 4.2, 7, 7.5, 7, 8.6],
                          # 'V6': [4, 8, 8.6, 6, 4.5, 5, 6.6],
                          # 'V7': [6, 7, 3.2, 9, 5.5, 8, 4.6],
                          # 'V8': [6, 4, 8.2, 6, 6.5, 7, 7.6],
                          # 'V9': [6, 7, 3.2, 9, 5.5, 8, 4.6]
                          })
    data2 = pd.DataFrame({'label': ['Feb', 'Jan', 'Mar', 'Apr', 'May', 'Aug', 'Seb'],
                          'V1': [5, 8, 4.3, 7, 8.2, 4.6, 8],
                          'V2': [6, 7, 8.2, 6, 4.5, 7, 8.6],
                          'V3': [7, 8, 7.2, 4, 3.5, 8, 8.9],
                          'V4': [5, 5, 8.2, 6, 4.5, 6, 9.6],
                          'V5': [6, 7, 4.2, 7, 7.5, 7, 8.6],
                          'V6': [4, 8, 8.6, 6, 4.5, 5, 6.6],
                          'V7': [6, 7, 3.2, 9, 5.5, 8, 4.6],
                          'V8': [6, 4, 8.2, 6, 6.5, 7, 7.6],
                          'V9': [6, 7, 3.2, 9, 5.5, 8, 4.6]
                          })
    # data0 = data1.copy()

    view =BarChart(data=data2)
    view.draw(style=3)
    plt.show()
    # view.draw(ax=view.ax,show_grid_type=None,show_gray_background=True,show_bar_edge=True)
    # plt.show()
    # view.draw(style=2,color_type=5,show_grid_type=None)
    # plt.show()
    # name_list = data0['label'].tolist()
    # num_list = data0.values.T.tolist()[1:]
    # print(name_list,type(name_list))
    # print(num_list,type(num_list))
    # print(math.ceil(max(max(num_list))+1))
    """
    data = pd.DataFrame({'X': ['2011年', '2012年', '2013年', '2014年', '2015年', '2016年', '2017年'],
                         'V1': [5800, 6020, 6300, 7100, 8400, 9050, 10700],
                         'V2': [5200, 5420, 5150, 5830, 5680, 5950, 6270],
                         'V3': [2600, 3320, 4050, 5130, 5380, 5450, 5870]})
    # 单子图绘制一次
    view_1 = LineChart(data, figsize=(8, 5))
    view_1.draw('style1', 'x', 'y', style=1, ax=view_1.ax)
    plt.show()

    view_2 = LineChart(data, figsize=(8, 5))
    view_2.draw('style2', 'x', 'y', style=4, ax=view_2.ax)
    plt.show()

    # 多个子图  参数subplots_shapezhi指定相同shape
    view_3 = LineChart(data, figsize=(8, 5), subplots_shape=(1, 2))
    view_4 = LineChart(data, figsize=(8, 5), subplots_shape=(1, 2))
    ax3 = view_3.ax[0]
    ax4 = view_3.ax[1]
    print(id(view_3))
    print(id(view_4))
    # 调用show函数显示
    # 在含有子图时保存图像需使最后一个子图save_fig=True 前面子图save_fig=False
    p3 = view_3.draw('style3', 'x', 'y', style=2, ax=ax3, save_fig=False)
    p4 = view_4.draw('style4', 'x', 'y', style=2, ax=ax4, save_fig=True)
    plt.show()
    """
    """
   
    # 饼状图单个图与子图绘制示例

    pie_data = pd.DataFrame({'label': ['A', 'B', 'C', 'D', '其他'],
                             'V1': [0.45, 0.25, 0.15, 0.05, 0.10]
                             })
    # ############### 单子图绘制一次  饼图风格1和2
    view_1 = PieChart(pie_data, figsize=(8, 5))
    view_1.draw(
        '2017年笔记本电脑市场份额',
        30,
        explode=None,
        show_legend=False,
        style=2,
        ax=view_1.ax)
    plt.show()
    ###############   # 单子图绘制一次  饼图风格3
    pie_data3 = pd.DataFrame({'label': ['A', 'B', 'C', 'D', '其他'],
                              'V1': [0.45, 0.25, 0.15, 0.05, 0.10],
                              'V2': [0.35, 0.34, 0.09, 0.07, 0.15]
                              })
    view_2 = PieChart(pie_data3, figsize=(8, 5))
    view_2.draw(
        '2017年笔记本电脑市场份额',
        30,
        explode=None,
        show_legend=False,
        style=3,
        ax=view_2.ax)
    plt.show()
    ############### 多子图绘制一次  饼图风格1和2
    pie_data = pd.DataFrame({'label': ['A', 'B', 'C', 'D', '其他'],
                             'V1': [0.45, 0.25, 0.15, 0.05, 0.10]
                             })
    view_3 = PieChart(pie_data, figsize=(8, 5), subplots_shape=(1, 2))
    view_4 = PieChart(pie_data, figsize=(8, 5), subplots_shape=(1, 2))
    ax3 = view_3.ax[0]
    ax4 = view_3.ax[1]
    view_3.draw(
        '2017年笔记本电脑市场份额',
        30,
        explode=None,
        show_legend=False,
        style=1,
        ax=ax3,
        save_fig=False)
    view_4.draw(
        '2017年笔记本电脑市场份额',
        30,
        explode=None,
        show_legend=False,
        style=2,
        ax=ax4,
        save_fig=True)
    plt.show()
     """
    """
    """
    # 柱状图示例
    """
    bar_data = pd.DataFrame({'label': ['Feb', 'Jan', 'Mar', 'Apr', 'May', 'Aug', 'Seb'],
                             'V1': [3.5, 3.6, 7.8, 6, 9, 6.3, 9.2],
                             'V2': [1.5, 1.6, 3.8, 3, 6, 2.3, 7.2],
                             })

    # view = BarChart(bar_data, figsize=(6, 3), dpi=150)
    # view.draw(
    #     '2017年笔记本电脑市场份额',
    #     20,
    #     style=1,
    #     ax=view.ax)
    # plt.show()
    # view.draw(
    #     '2017年笔记本电脑市场份额',
    #     20,
    #     style=2,
    #     ax=view.ax)
    # plt.show()
    # view.draw(
    #     '2017年笔记本电脑市场份额',
    #     20,
    #     style=3,
    #     ax=view.ax)
    # plt.show()
    #########  多个子图 可以设置子图显示不同图
    pie_data = pd.DataFrame({'label': ['A', 'B', 'C', 'D', '其他'],
                             'V1': [0.45, 0.25, 0.15, 0.05, 0.10]
                             })
    view_3 = PieChart(pie_data, figsize=(8, 5), subplots_shape=(1, 2))
    view_4 = BarChart(pie_data, figsize=(8, 5), subplots_shape=(1, 2))
    ax3 = view_3.ax[0]
    ax4 = view_3.ax[1]
    view_3.draw(
        '2017年笔记本电脑市场份额',
        30,
        style=1,
        ax=ax3)
    view_4.draw(
        '2017年笔记本电脑市场份额',
        30,
        style=1,
        ax=ax4)
    plt.show()
    """

    # 箱型图示例
    """
    box_data = pd.DataFrame({ 'V1': np.random.normal(0,1,100),
                     'V2': np.random.normal(0,2,100),
                     'V3': np.random.normal(0,3,100),
                     })
    ### 单个子图
    view = BoxChart(box_data,figsize = (6,3) ,dpi =150)
    view.draw(
        '2017年笔记本电脑市场份额',
        30,
        style=1,
        ax=view.ax)
    plt.show()

    # ### 多个子图
    view_3 = BoxChart(box_data, figsize=(8, 5), subplots_shape=(1, 2))
    view_4 = BoxChart(box_data, figsize=(8, 5), subplots_shape=(1, 2))
    ax3 = view_3.ax[0]
    ax4 = view_3.ax[1]
    view_3.draw(
        '2017年笔记本电脑市场份额',
        30,
        style=1,
        ax=ax3)
    view_4.draw(
        '2017年笔记本电脑市场份额',
        30,
        style=1,
        ax=ax4)
    plt.show()

    """

    # 热力图示例
    """
    # 单个子图
    np.random.seed(0)
    data = pd.DataFrame({'month': ['一月', '二月', '三月', '四月', '五月'],
                         '2016': [1350, 1500, 1400, 1552, 1632],
                         '2017': [1350, 1400, 1500, 1852, 1562],
                         '2018': [1000, 1200, 1300, 1500, 1452],
                         '2019': [1200, 1100, 1500, 1400, 1552],
                         '2020': [1300, 1200, 1400, 1500, 1852],
                         '2021': [1500, 1300, 1400, 1200, 1542],
                         '2022': [1532, 1100, 1500, 1400, 1552]})
    # view = HeatmapChart(data,figsize = (6,3) ,dpi =150)
    # view.draw(
    #     '2016-2019年XX热力图',
    #     30,
    #     style=1,
    #     ax=view.ax)
    # plt.show()
    ### 多个子图
    view_3 = HeatmapChart(data, figsize=(8, 8), subplots_shape=(1, 2))
    view_4 = HeatmapChart(data, figsize=(8, 8), subplots_shape=(1, 2))
    ax3 = view_3.ax[0]
    ax4 = view_3.ax[1]
    view_3.draw(
        '2016-2019年XX热力图',
        30,
        style=1,
        ax=ax3)
    view_4.draw(
        '2016-2019年XX热力图',
        30,
        style=1,
        ax=ax4)
    plt.show()
    """
