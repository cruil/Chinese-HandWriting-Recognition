#python3
#coding:utf-8

import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
import cv2
from PIL import Image


#创建一个logging对象，参数为将要返回的日志期的抿成标识
logger = logging.getLogger('Training the chinese write handling char recognition')
#设置日志器将会处理的日志消息的最低严重级别
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#将日志消息发送到stream，如std.out,std.err或任何file-like对象
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#为logger添加一个haddler对象
logger.addHandler(ch)

#输入参数解析
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

#修改了char_size
tf.app.flags.DEFINE_integer('charset_size', 20, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 12002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 50, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 2000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
#tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        #Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        #遍历训练集所有图像的路径，存储在image_names内
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        #随机化
        random.shuffle(self.image_names)
        #得到对应的图片文件名的数字后标
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        #镜像变换
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        #图像亮度变化
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        #对比度变化
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images
    # batch的生成
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        #numpy array 转 tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        #生成一个输入队列，num_epochs表示可以遍历列表的最大轮数，达到最大轮数，会报OutOfRange，默认打乱顺序
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        #将png图像进行解码，得到图像对应的三维矩阵，再将数据类型转化为实数用于后面图像处理
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        #对图像进行随机翻转，随即改变亮度，随机该百年对比度，利用有限的训练数据，提高模型的健壮性
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        #训练图片大小不固定，为了适应神经网络的输入节点，先统一图像的大小
        images = tf.image.resize_images(images, new_size)
        #将多个输入样例组合成一个batch。生成一个队列，每次出对即得到一个batch的样例，capacity为队列的大小min_after_dequeue限定了队列中最少元素的个数
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        #每次返回一个batch的image,label
        return image_batch, label_batch


def build_graph(top_k):
    # with tf.device('/cpu:0'):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    #网络结构
    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')
    conv_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv4')
    max_pool_4 = slim.max_pool2d(conv_4, [2, 2], [2, 2], padding='SAME')
    conv_5 = slim.conv2d(max_pool_4, 1024, [3, 3], padding='SAME', scope='conv5')
    max_pool_5 = slim.max_pool2d(conv_5, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_5)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
    #最终输出有FLAGS.charset_size个节点，就是有这么多类
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='fc2')
    # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    #softmax后的交叉熵损失（多分类问题）
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    #指数衰减学习率（初始学习率，global_step，衰减速度，衰减系数，staircase）   每训练2000轮，学习率乘以0.96
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)
    #记录信息
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    #保存所有summary
    merged_summary_op = tf.summary.merge_all()
    #结果中最大的k个，及其索引
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    #输出结果前k个值（序号）是否包含target中的值，若包含则返回true。之后转化为实数（1或0），求均值，得正确率
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='./data/train/')
    #用到测试集是因为训练过程中，每过固定轮数，会进行一次测试
    test_feeder = DataIterator(data_dir='./data/test/')
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=128, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=128)
        graph = build_graph(top_k=1)
        sess.run(tf.global_variables_initializer())
        #设置多线程协调器
        #声明一个tf.train.Coordinator类来协同多个线程
        coord = tf.train.Coordinator()
        #启动所有线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        #将计算图写入给定的路径
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        #可以从某个step下的模型继续训练
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8}
                #执行
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)

                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                #记录日志。步数，用时，损失值
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                #每隔固定轮做一次测试，日志记录正确率
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('===============Eval a batch=======================')
                #每隔固定轮保存一下
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            #达到最大训练迭代数的时候清理关闭线程
            coord.request_stop()
        coord.join(threads)


def validation():
    print('Begin validation')
    test_feeder = DataIterator(data_dir='./data/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=128, num_epochs=1)
        graph = build_graph(3)

        sess.run(tf.global_variables_initializer())
        #初始化局部变量（GraphKeys.LOCAL_VARIABLE），被添加入图，但未被存储的变量
        sess.run(tf.local_variables_initializer())  #initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0}
                #依次获得标签列表，预测可能性最大的前k个值，及对应的索引，正确率，前k个预测值中有正确答案的可能性
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                
                #转换成列表                                                       
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            #求均值
            acc_top_1 = acc_top_1 * 128 / test_feeder.size
            acc_top_k = acc_top_k * 128 / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}

class StrToBytes:  
    def __init__(self, fileobj):  
        self.fileobj = fileobj  
    def read(self, size):  
        return self.fileobj.read(size).encode()  
    def readline(self, size=-1):  
        return self.fileobj.readline(size).encode()

#获取汉字label映射表
def get_label_dict():
    #f=open('./chinese_labels','r')
    #label_dict = pickle.load(f)
    #f.close()
    with open('./chinese_labels', 'r') as data_file:
        #从文件中的对象序列化读出
        label_dict = pickle.load(StrToBytes(data_file))
        return label_dict

#获待预测图像文件夹内的图像名字
def get_file_list(path):
    list_name=[]
    #获得./tmp文件夹下所有文件的文件名列表
    files = os.listdir(path)
    files.sort()
    #得到所有测试文件的路径的列表
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name

def inference1(name_list):
    print('inference1')
    image_set=[]
    #对每张图进行尺寸标准化和归一化
    for image in name_list:
        #读取为灰度图
        temp_image = Image.open(image).convert('L')
        #调整为适应神经网络输入的尺寸，并抗锯齿
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        #将输入调整为[0,1]        
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)

    #allow_soft_placement 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        logger.info('========start inference============')
        #images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        #Pass a shadow label 0. This label will not affect the computation graph.
           
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        #自动获取最后一次保存的模型
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:       
            saver.restore(sess, ckpt)
        val_list=[]
        idx_list=[]
        #预测每一张图
        for item in image_set:
            temp_image = item
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0})
            #前k个预测答案
            val_list.append(predict_val)
            #前k个预测答案在训练集对应的索引，可通过这个索引找到对应的是哪个汉字
            idx_list.append(predict_index)
        #注意加这一句，如果单独运行chinese_rec.py，这一句是不需要的。因为只运行一次inference1()
        #但如果运行QTGUI_Chinese_rec.py,每选一次文件，都要运行一次inference1()
        #每次都要构建一次计算图，这样会出错
    tf.reset_default_graph()
    #return predict_val, predict_index
    #找bug
    #print(idx_list)
    return val_list,idx_list

def inference2():
    print('inference2')
    '''
    global final_predict_valinf2
    global final_predict_indexinf2
    global imageinf2
    '''
    capture = cv2.VideoCapture(0) #-> 0->1 for usb camera
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        #loader = tf.train.import_meta_graph('./model.meta')
        #loader.restore(sess,'./model')
        logger.info('========start inference============')
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
        while(True):
            #截图文件夹里只保存一张图就可以了，每次先把之前有的文件删掉
            label_dict = get_label_dict()
            ret, frame = capture.read()
            show_img = frame.copy()
            sp=show_img.shape
            cv2.rectangle(show_img, (100,100), (200, 200), (0, 0, 255), 5)

            crop_img = frame[100:200, 100:200]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = crop_img.reshape(100,100)

            show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2GRAY)
            show_img = show_img.reshape(480,640)
            
            #增加点对比度 去掉汉字外的干扰颜色
            for i in range(100,200):
                for j in range(100,200):
                    if(show_img[i,j]>80 and show_img[i,j] <= 130):
                        show_img[i,j] += 125
                    elif(show_img[i,j]>130):
                        show_img[i,j] = 245
            
            cv2.imshow('frame', show_img)
            
            k = cv2.waitKey(1) 
        
            if k == ord('s'):
                #增加点对比度
                for i in range(100):
                    for j in range(100):
                        if(crop_img[i,j]>80 and crop_img[i,j]<=130):
                            crop_img[i,j] += 125
                        elif(crop_img[i,j]>130):
                            crop_img[i,j] = 245   

                #将数组转化为图片，这一步找了好久。。。  这一句要放在保存图片之后，不然图片保存不了，出错
                crop_img = Image.fromarray(crop_img)       
                temp_image = crop_img.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
                temp_image = np.asarray(temp_image) / 255.0
                temp_image = temp_image.reshape([-1, 64, 64, 1])
                val_list=[]
                idx_list=[] 
                predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                          feed_dict={graph['images']: temp_image,
                                                     graph['keep_prob']: 1.0})
                #前k个预测答案
                val_list.append(predict_val)
                #前k个预测答案在训练集对应的索引，可通过这个索引找到对应的是哪个汉字
                idx_list.append(predict_index)
                final_reco_text =[]
                #预测结果的index
                candidate1 = idx_list[0][0][0]
                candidate2 = idx_list[0][0][1]
                candidate3 = idx_list[0][0][2]
                '''
                print(candidate1)
                print(candidate2)
                print(candidate3)
                print(idx_list[0])
                print(val_list[0])
                print(label_dict[int(candidate1)])
                print(label_dict[int(candidate2)])
                print(label_dict[int(candidate3)])
                '''
                #记录预测结果
                final_reco_text.append(label_dict[int(candidate1)])
                #记录日志
                '''
                logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
                        label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],idx_list[i],val_list[i]))
                '''
                logger.info('predict: {0} {1} {2}; predict index {3} predict_val {4}'.format(
                        label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],idx_list[0],val_list[0]))
                print ('=====================OCR RESULT=======================\n')
                #打印出所有识别出来的结果（取top 1）
                for i in range(len(final_reco_text)):
                    print(final_reco_text[i],)
            elif k == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
    tf.reset_default_graph()

def inference3():
    print('inference3')
    '''
    global final_predict_valinf2
    global final_predict_indexinf2
    global imageinf2
    '''
    capture = cv2.VideoCapture(0) #-> 0->1 for usb camera
    imageinf2_path = "./imageinf2_path"
    val_list=[]
    idx_list=[]
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        #loader = tf.train.import_meta_graph('./model.meta')
        #loader.restore(sess,'./model')
        logger.info('========start inference============')
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
        while(True):
            #截图文件夹里只保存一张图就可以了，每次先把之前有的文件删掉
            if os.path.exists("./imageinf2_path/imageinf2.png"):
                os.remove("./imageinf2_path/imageinf2.png")
            label_dict = get_label_dict()
            ret, frame = capture.read()
            show_img = frame.copy()
            cv2.rectangle(show_img, (100,100), (200, 200), (0, 0, 255), 5)

            crop_img = frame[100:200, 100:200]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = crop_img.reshape(100,100)

            show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2GRAY)
            show_img = show_img.reshape(480,640)
            
            #增加点对比度
            for i in range(100,200):
                for j in range(100,200):
                    if(show_img[i,j]>80 and show_img[i,j] <= 130):
                        show_img[i,j] += 125
                    elif(show_img[i,j]>130):
                        show_img[i,j] = 245
            cv2.imshow('fram e', show_img)

            k = cv2.waitKey(1) 
        
            if k == ord('s'):
                #增加点对比度
                for i in range(100):
                    for j in range(100):
                        if(crop_img[i,j]>80 and crop_img[i,j]<=130):
                            crop_img[i,j] += 125
                        elif(crop_img[i,j]>130):
                            crop_img[i,j] = 245
                #先把截到的图存下来，然后返回给gui，直接返回我不太会，所有先保存成文件再返回
                file_name=("./imageinf2_path/imageinf2.png")
                cv2.imwrite(file_name, crop_img)
                #将数组转化为图片，这一步找了好久。。。  这一句要放在保存图片之后，不然图片保存不了，出错
                crop_img = Image.fromarray(crop_img)       
                temp_image = crop_img.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
                temp_image = np.asarray(temp_image) / 255.0
                temp_image = temp_image.reshape([-1, 64, 64, 1])
                predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                          feed_dict={graph['images']: temp_image,
                                                     graph['keep_prob']: 1.0})
                #前k个预测答案
                val_list.append(predict_val)
                #前k个预测答案在训练集对应的索引，可通过这个索引找到对应的是哪个汉字
                idx_list.append(predict_index)
                final_reco_text =[]
                #预测结果的index
                candidate1 = idx_list[0][0][0]
                candidate2 = idx_list[0][0][1]
                candidate3 = idx_list[0][0][2]
                '''
                print(candidate1)
                print(candidate2)
                print(candidate3)
                print(idx_list[0])
                print(val_list[0])
                print(label_dict[int(candidate1)])
                print(label_dict[int(candidate2)])
                print(label_dict[int(candidate3)])
                '''
                #记录预测结果
                final_reco_text.append(label_dict[int(candidate1)])
                #记录日志
                '''
                logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
                        label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],idx_list[i],val_list[i]))
                '''
                logger.info('predict: {0} {1} {2}; predict index {3} predict_val {4}'.format(
                        label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],idx_list[0],val_list[0]))
                print ('=====================OCR RESULT=======================\n')
                #打印出所有识别出来的结果（取top 1）
                for i in range(len(final_reco_text)):
                    print(final_reco_text[i],)
                break
            elif k == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
    tf.reset_default_graph()
    return val_list,idx_list


def main(_):
    print(FLAGS.mode)
    #根据命令行给的参数，来进行train/validation/inference
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            #将验证结果存入打开的f中
            pickle.dump(dct, f)
        #记录日志
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference1':
        label_dict = get_label_dict()
        #获得测试用图片的路径列表
        name_list = get_file_list('./tmp')
        final_predict_val, final_predict_index = inference1(name_list)
        #image_path = './tmp/128.jpg'
        #final_predict_val, final_predict_index = inference(image_path)
        #logger.info('the result info label {0} predict index {1} predict_val {2}'.format(final_predict_index[0][0], final_predict_index,final_predict_val))
        #logger.info('|{0},{1:.0%}|{2},{3:.0%}|{4},{5:.0%}|'.format(label_dict[int(final_predict_index[0][0])],final_predict_val[0][0],label_dict[int(final_predict_index[0][1])],final_predict_val[0][1],label_dict[int(final_predict_index[0][2])],final_predict_val[0][2]))
        
        #存储最后识别出来的文字串
        final_reco_text =[]
        print(final_predict_index)
        #给出top3预测，candidate1是概率最高的预测
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0][0]
            candidate2 = final_predict_index[i][0][1]
            candidate3 = final_predict_index[i][0][2]
            #找到对应的汉字，预测概率前三
            final_reco_text.append(label_dict[int(candidate1)])
            #记录日志
            logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
                label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],final_predict_index[i],final_predict_val[i]))
        print ('=====================OCR RESULT=======================\n')
        #打印出所有识别出来的结果（取top 1）
        for i in range(len(final_reco_text)):
           print(final_reco_text[i],)
    elif FLAGS.mode == 'inference2':
        #因为要实时识别，所以全部封装函数里面
        inference2()

if __name__ == "__main__":
    tf.app.run()
