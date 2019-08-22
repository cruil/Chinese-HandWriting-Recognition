# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QMainWindow, QLineEdit, QAction, QFileDialog, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel,QWidget,QMessageBox,QFrame)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QPalette, QBrush, QColor
from PyQt5.QtCore import Qt
import sys
import chinese_rec
import pickle

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        #示例对应的三个可能性
        self.ansPredict_1 = 100
        self.ansPredict_2 = 0
        self.ansPredict_3 = 0
        #示例所用图片文件路径
        self.myFileStr = './data/test/00999/223400.png'  # 这是inference的输入
        #示例答案
        self.ans = '帆'  # 这是inference的结果
        #示例路径显示在行编辑器里
        self.LineEdit = QLineEdit(self.myFileStr)
        self.okButton = QPushButton("OK")
        #“OK”按钮按下关联事件
        self.okButton.clicked.connect(self.showAns1)
        self.okButton.setFixedSize(60,23)
        self.openButton = QPushButton("...")
        #关联事件，显示文件对话框
        self.openButton.clicked.connect(self.showDialog)
        #按钮大小
        self.openButton.setFixedSize(23,23)

        #用摄像头取字
        self.camButton = QPushButton("摄像头")
        self.camButton.setFixedSize(60,23)
        self.camButton.clicked.connect(self.showAns2)
        #用于显示图像，图像来源于路径self.myFileStr
        pixmap = QPixmap(self.myFileStr,'wr')
        #定义标签，用来显示文件原始图
        self.graphLabel = QLabel()
        self.graphLabel.setPixmap(pixmap)
        self.graphLabel.setScaledContents(True)
        self.graphLabel.setFixedSize(150,150)
        #显示在标签中央
        self.graphLabel.setAlignment(Qt.AlignCenter)

        #说明性的文字标签
        self.graphText = QLabel()
        self.graphText.setText('原始图片：')
        self.graphText.setFont(QFont("ZhongSong", 20, QFont.Bold))

        #结果1
        #用来显示答案
        self.ansLabel_1 = QLabel(self.ans)
        self.ansLabel_1.setFont(QFont("ZhongSong", 100, QFont.Bold))
        # palette = QPalette()
        # palette.setBrush(self.backgroundRole(),QBrush(pixmap))
        # self.ansLabel_1.setPalette(palette)
        #self.ansLabel_1.setStyleSheet("border:2px")
        # palette_1 = QPalette()
        # palette_1.setColor(self.backgroundRole(), QColor(0, 0, 0))
        # self.setPalette(palette_1)
        #self.ansLabel_1.setScaledContents(True)
        self.ansLabel_1.setFixedSize(150,150)
        self.ansLabel_1.setAlignment(Qt.AlignCenter)
        # palette_2 = QPalette()
        # palette_2.setColor(self.backgroundRole(),QColor(255,255,255))
        # self.ansLabel_1.setPalette(palette_2)
        #背景图
        self.ansLabel_1.setStyleSheet('QLabel{border-image:url(background.jpg);}')

        #答案结果和可能性 两个说明性文字标签
        self.ansText_11 = QLabel()
        self.ansText_11.setText("识别结果1：")
        self.ansText_11.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_12 = QLabel()
        #可能性后面可能性是变量
        self.ansText_12.setText("可能性：{0}%".format(self.ansPredict_1))
        self.ansText_12.setFont(QFont("ZhongSong", 20, QFont.Bold))
        #纵向布局
        self.vbox_1 = QVBoxLayout()
        self.vbox_1.addWidget(self.ansText_11)
        self.vbox_1.addWidget(self.ansText_12)

        #结果2
        self.ansLabel_2 = QLabel()
        self.ansLabel_2.setFixedSize(150, 150)
        # self.ansLabel_2.setScaledContents(True)
        self.ansLabel_2.setAlignment(Qt.AlignCenter)
        self.ansLabel_2.setStyleSheet('border-image:url(background.jpg)')

        self.ansText_21 = QLabel()
        self.ansText_21.setText("识别结果2：")
        self.ansText_21.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_22 = QLabel()
        #self.ansPredict_1 = 100
        self.ansText_22.setText("可能性：{0}%".format(self.ansPredict_2))
        self.ansText_22.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.vbox_2 = QVBoxLayout()
        self.vbox_2.addWidget(self.ansText_21)
        self.vbox_2.addWidget(self.ansText_22)

        #结果3
        self.ansLabel_3 = QLabel()
        #self.ansLabel_3.setFont(QFont("ZhongSong", 100, QFont.Bold))
        self.ansLabel_3.setFixedSize(150, 150)
        self.ansLabel_3.setAlignment(Qt.AlignCenter)
        self.ansLabel_3.setStyleSheet("QLabel{border-image:url(background.jpg)}")

        self.ansText_31 = QLabel()
        self.ansText_31.setText("识别结果3：")
        self.ansText_31.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_32 = QLabel()
        #self.ansPredict_1 = 100
        self.ansText_32.setText("可能性：{0}%".format(self.ansPredict_3))
        self.ansText_32.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.vbox_3 = QVBoxLayout()
        self.vbox_3.addWidget(self.ansText_31)
        self.vbox_3.addWidget(self.ansText_32)

        #横向布局
        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(self.LineEdit)
        hbox_1.addWidget(self.openButton)
        hbox_1.addWidget(self.camButton)
        # hbox_1.addStretch(1)
        hbox_1.addWidget(self.okButton)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(self.graphText)
        #hbox_2.addSpacing(30)
        hbox_2.addWidget(self.graphLabel)
        #hbox_2.addSpacing(30)

        hbox_3 = QHBoxLayout()
        hbox_3.addLayout(self.vbox_1)
        hbox_3.addWidget(self.ansLabel_1)
        # hbox_2.addStretch(1)
        #hbox_2.addWidget(self.ansLabel)

        hbox_4 = QHBoxLayout()
        hbox_4.addLayout(self.vbox_2)
        hbox_4.addWidget(self.ansLabel_2)

        hbox_5 = QHBoxLayout()
        hbox_5.addLayout(self.vbox_3)
        hbox_5.addWidget(self.ansLabel_3)

        #将几个横向布局加入纵向布局
        vbox = QVBoxLayout()
        vbox.addLayout(hbox_1)
        vbox.addSpacing(30)
        vbox.addLayout(hbox_2)
        #vbox.addSpacing(30)
        vbox.addLayout(hbox_3)
        vbox.addLayout(hbox_4)
        vbox.addLayout(hbox_5)

        self.setLayout(vbox)

        self.setGeometry(550,100,294,800)
        self.setWindowTitle("手写汉字识别")
        self.setWindowIcon(QIcon('feather-pencil-512.png'))
        self.show()

    def showDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', './data/test/')

        if fname[0]:
            # print(fname)
            f = open(fname[0], 'r')
            self.myFileStr = fname[0]
            self.LineEdit.setText(fname[0])
            pixmap = QPixmap(self.myFileStr)
            self.graphLabel.setPixmap(pixmap)
            print(self.myFileStr)

    def showAns1(self):
        # QApplication.processEvents()
        label_dict = chinese_rec.get_label_dict()
        final_predict_val, final_predict_index = chinese_rec.inference1([self.myFileStr])
        # logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,final_predict_val))
        # print(final_predict_index)
        # print(final_predict_index[0][0])

        ansIndex_1=final_predict_index[0][0][0]
        ansIndex_2=final_predict_index[0][0][1]
        ansIndex_3=final_predict_index[0][0][2]
        self.ansPredict_1 = final_predict_val[0][0][0]*100
        self.ansPredict_2 = final_predict_val[0][0][1]*100
        self.ansPredict_3 = final_predict_val[0][0][2]*100
        self.ans_1 = label_dict[int(ansIndex_1)]
        self.ans_2 = label_dict[int(ansIndex_2)]
        self.ans_3 = label_dict[int(ansIndex_3)]

        #标签里面的文字
        self.ansLabel_1.setText(self.ans_1)
        self.ansLabel_1.setFont(QFont("ZhongSong", 100, QFont.Bold))
        # self.ansLabel_1.setFrameShape(QFrame.Panel)
        # self.ansLabel_1.setFrameShadow(QFrame.Sunken)
        # self.ansLabel_1.setLineWidth(3)
        self.ansLabel_2.setText(self.ans_2)
        self.ansLabel_2.setFont(QFont("ZhongSong", 100, QFont.Bold))
        self.ansLabel_3.setText(self.ans_3)
        self.ansLabel_3.setFont(QFont("ZhongSong", 100, QFont.Bold))
        #可能性标签里面的文字
        self.ansText_12.setText("可能性：{:.2f}%".format(self.ansPredict_1))
        self.ansText_22.setText("可能性：{:.2f}%".format(self.ansPredict_2))
        self.ansText_32.setText("可能性：{:.2f}%".format(self.ansPredict_3))
        #return self.ans
        # for key, value in charDict.items():
        #     print(key)
        #     if value == final_predict_index[0]:
        #
        #         self.ans= key
        #         self.ansLabel.setText(self.ans)

    def showAns2(self):
        # QApplication.processEvents()
        final_predict_valinf2,final_predict_indexinf2=chinese_rec.inference3()
        file_name=("./imageinf2_path/imageinf2.png")
        pixmap = QPixmap(file_name)
        self.graphLabel.setPixmap(pixmap)
        label_dict = chinese_rec.get_label_dict()

        ansIndex_1=final_predict_indexinf2[0][0][0]
        ansIndex_2=final_predict_indexinf2[0][0][1]
        ansIndex_3=final_predict_indexinf2[0][0][2]
        self.ansPredict_1 = final_predict_valinf2[0][0][0]*100
        self.ansPredict_2 = final_predict_valinf2[0][0][1]*100
        self.ansPredict_3 = final_predict_valinf2[0][0][2]*100
        self.ans_1 = label_dict[int(ansIndex_1)]
        self.ans_2 = label_dict[int(ansIndex_2)]
        self.ans_3 = label_dict[int(ansIndex_3)]
        
        #标签里面的文字
        self.ansLabel_1.setText(self.ans_1)
        self.ansLabel_1.setFont(QFont("ZhongSong", 100, QFont.Bold))
        # self.ansLabel_1.setFrameShape(QFrame.Panel)
        # self.ansLabel_1.setFrameShadow(QFrame.Sunken)
        # self.ansLabel_1.setLineWidth(3)
        self.ansLabel_2.setText(self.ans_2)
        self.ansLabel_2.setFont(QFont("ZhongSong", 100, QFont.Bold))
        self.ansLabel_3.setText(self.ans_3)
        self.ansLabel_3.setFont(QFont("ZhongSong", 100, QFont.Bold))
        #可能性标签里面的文字
        self.ansText_12.setText("可能性：{:.2f}%".format(self.ansPredict_1))
        self.ansText_22.setText("可能性：{:.2f}%".format(self.ansPredict_2))
        self.ansText_32.setText("可能性：{:.2f}%".format(self.ansPredict_3))

# class ansWindow(QWidget):
#     def __init__(self,ans):
#         super().__init__()
#         self.resize(200,200)
#         self.ans=ans
#     def handle_click(self):
#         self.ansLabel = QLabel(self.ans)
#         self.ansLabel.setScaledContents(True)
#         self.ansLabel.setFont(QFont("ZhongSong", 100, QFont.Bold))
#         self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyWindow()
    #ans = mainWindow.showAns()
    #popWindow = ansWindow(ans)
    #mainWindow.okButton.clicked.connect(popWindow.handle_click)
    # mainWindow.show()
    sys.exit(app.exec_())