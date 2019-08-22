# Chinese-HandWriting-Recognition
双学位毕业设计手写汉字识别，用PyQt做了个界面
（这个基本上是抄的别人的，把别人原有的拿来在主体框架上改了改，加了点功能，让我自己写还写不出来，把别人的代码弄懂了感觉也还不错）
anaconda python3.6.9 tensorflow-gpu1.13.1 (PyQt5 5.13)

数据集来自于中科院自动化研究所，总共3755个汉字
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip 
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
上面的没法直接用，具体可以看看这个https://zhuanlan.zhihu.com/p/24698483
这个网盘里面的可以下下来直接用
https://pan.baidu.com/s/1o84jIrg 
下载下来有一个test的压缩文件，还有好几个数字编号的文件夹是训练集，训练集解压一个就够了，我只解压了一个够用了
还有一个文件时编码汉字的吧，我不太了解，是为了后面编码汉字输出汉字用的就是本项目里面用到的 chinese_labels文件

#运行：
训练：
anaconda进入Chinese-HandWriting-Recognition文件夹,激活tensorflow-gpu,
python chinese_rec.py --mode=train --max_steps=16002 --eval_steps=100 --save_steps=500，通过mode的值来指定训练还是验证，
后面三个参数是训练的轮数，每隔多少次验证一下，每个多少次保存一下模型。可以自己调整（）
还有，不一定要把3755个字全部训练了，可以改变chinese_rec.py里面charset_size的值，来只使用一部分数据，这样可以节省点时间，机器性能不好也可以缓解一下
验证：
python chinese_rec.py --mode=validation
推理退断（开始识字了）：
chinese_rec里面有三个inference函数：
inference1()是识别指定汉字图片的，命令行：python chinese_rec.py --mode=inference1，把要是别的汉字png格式图片放到./tmp文件夹下就可以啦
inference2()是通过摄像头识字，将写好的字通过摄像头放在画面中红色的方框内，按下“s”键，进行识别，按下“q”键退出。
  这里摄像头用的是笔记本自带的摄像头，如果使用usb摄像头，将capture = cv2.VideoCapture(0)改为capture = cv2.VideoCapture(1)
  命令行：python chinese_rec.py --mode=inference1
inference3()是给那个GUI用的，功能和inference2()是一样的，但这个有返回值，不能连续识字，识一个字摄像头就退出了。

为什么有了inference2,还要写inference3,因为我比较菜，GUI是后面加上去的，为了省事，就用了两个。

如果直接运行python QTGUI_chinese_rec.py 可以把之前在命令行的东西搞到一个界面里，点击“...”按钮选择要是别的png图片，点击ok,输出结果。
点击“摄像头”按钮，打开摄像头，再按下“s”键，输出识别结果，摄像头关闭。下一次需要再点击“摄像头”

参考：
https://github.com/Mignet/chinese-write-handling-char-recognition
https://zhuanlan.zhihu.com/p/24698483
https://www.bilibili.com/video/av24682059?from=search&seid=9325737414826528610
https://www.bilibili.com/video/av51063658?from=search&seid=9325737414826528610
