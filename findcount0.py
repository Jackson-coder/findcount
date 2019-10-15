from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

im = Image.open('/home/linyihong/findcount/presolve/test.png') #读取的图片所在路径，注意是28*28像素
plt.imshow(im)  #显示需要识别的图片
plt.show()
im = im.convert('L')
tv = list(im.getdata()) 
result = [(255-x)*1.0/255.0 for x in tv] 

x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])
#定义一个获取卷积核的函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#定义一个获取偏置值的函数
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义一个卷积函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME")

#定义一个池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
    
x = tf.placeholder(shape=[None,28*28],dtype=tf.float32)
lable = tf.placeholder(shape=[None,10],dtype=tf.float32)

x_image = tf.reshape(x,[-1,28,28,1])

#第一个卷积层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#14*14*32

#第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#7*7*64

#全连接层，输出为1024维向量
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

#把1024维向量转换成10维，对应10个类别
W_fc2 = weight_variable([1024,10])
b_fc2 = weight_variable([10])
y_conv = tf.matmul(h_fc1,W_fc2)+b_fc2

#直接使用tf.nn.softmax_cross_entropy_with_logits直接计算交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable,logits=y_conv))
#定义train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(lable,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./model.ckpt") #使用模型，参数和之前的代码保持一致

    start = time()

    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    
    stop = time()

    print('识别结果:')
    print(predint[0])
    
    print(str(stop-start) +"秒")