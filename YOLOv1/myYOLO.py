import numpy as np
import tensorflow as tf
import cv2


'''
def load_train_data():
    emmmm没有数据用
    images = n * 448 * 448 * 3 的张量
    targets = n * 7 * 7 * 25 的张量
    return images, targets
'''


class Yolo(object):
    def __init__(self):
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
        self.C = len(self.classes) # number of classes
        self.S = 7  # cell size
        self.B = 2  # boxes_per_cell

        self.sess = tf.Session()
        self.build_net()

        # 检测时将网络的输出翻译成人话要用到的东西们
        self.threshold = 0.2  # confidence scores threshold
        self.iou_threshold = 0.5

        self.idx1 = self.S*self.S*self.C
        self.idx2 = self.idx1 + self.S*self.S*self.B
        
        # offset for box center (top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                              [self.B, self.S, self.S]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        # 训练要用到的东西们
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1
        self.lambda_class = 1
        
        self.learning_rate = 0.02
        

    def build_net(self):
        print("Start to build the network ...")
        self.img_input = tf.placeholder(tf.float32, [None, 448, 448, 3])
        out = self.conv_layer(self.img_input, 64, 7, 2)
        out = self.maxpool_layer(out, 2, 2)
        out = self.conv_layer(out, 192, 3, 1)
        out = self.maxpool_layer(out, 2, 2)
        out = self.conv_layer(out, 128, 1, 1)
        out = self.conv_layer(out, 256, 3, 1)
        out = self.conv_layer(out, 256, 1, 1)
        out = self.conv_layer(out, 512, 3, 1)
        out = self.maxpool_layer(out, 2, 2)
        out = self.conv_layer(out, 256, 1, 1)
        out = self.conv_layer(out, 512, 3, 1)
        out = self.conv_layer(out, 256, 1, 1)
        out = self.conv_layer(out, 512, 3, 1)
        out = self.conv_layer(out, 256, 1, 1)
        out = self.conv_layer(out, 512, 3, 1)
        out = self.conv_layer(out, 256, 1, 1)
        out = self.conv_layer(out, 512, 3, 1)
        out = self.conv_layer(out, 512, 1, 1)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.maxpool_layer(out, 2, 2)
        out = self.conv_layer(out, 512, 1, 1)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.conv_layer(out, 512, 1, 1)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.conv_layer(out, 1024, 3, 2)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.conv_layer(out, 1024, 3, 1)
        out = self.flatten(out)
        out = self.fc_layer(out, 512, activation=self.leak_relu)
        out = self.fc_layer(out, 4096, activation=self.leak_relu)
        out = self.fc_layer(out, self.S*self.S*(self.C+5*self.B))
        self.net_output = out


    def leak_relu(self, x, alpha=0.1): # 激活函数
        return tf.maximum(alpha * x, x)


    def conv_layer(self, input, num_filters, filter_size, stride): # 卷积层
        in_channels = input.get_shape().as_list()[-1] # 输入通道数
        filters = tf.Variable(tf.truncated_normal([filter_size, filter_size, # 随机正态分布初始化的卷积核们
                                                  in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters,])) 

        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        input_padded = tf.pad(input, pad_mat)
        
        output = tf.nn.conv2d(input_padded, filters, strides=[1, stride, stride, 1], padding="VALID")
        output = self.leak_relu(tf.nn.bias_add(output, bias))

        return output


    def maxpool_layer(self, input, pool_size, stride): # 最大池化层
        output = tf.nn.max_pool(input, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")
        
        return output


    def flatten(self, input):
        input = tf.transpose(input, [0, 3, 1, 2])  # channle first mode
        nums = np.product(input.get_shape().as_list()[1:]) # 共7*7*1024个
        return tf.reshape(input, [-1, nums])

    
    def fc_layer(self, input, num_out, activation=None): # 全连接层
        num_in = input.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out,]))

        output = tf.nn.xw_plus_b(input, weight, bias)
        if activation:
            output = activation(output)
            
        return output


    def detect(self, image_file, weights_file):
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file) # 加载训练好的权重
        # read image
        image = cv2.imread(image_file) # imread读进BGR模式
        img_h, img_w, _ = image.shape # 图片原本的大小
        img_input = cv2.resize(image, (448, 448)) # 统一图像大小为448*448
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_input = (img_input / 255.0) * 2.0 - 1.0
        img_input = np.reshape(img_input, (1, 448, 448, 3))
        
        net_output = self.sess.run(self.net_output, feed_dict={self.img_input: img_input})[0]
        predict_boxes = self.interpret_output(net_output, img_h, img_w) # 预测出的框们(class, x, y, w, h, score)
        self.show_results(image, predict_boxes)


    def interpret_output(self, output, img_h, img_w):
        class_probs = np.reshape(output[:self.idx1], [self.S, self.S, self.C]) # class prediction
        confs = np.reshape(output[self.idx1:self.idx2], [self.S, self.S, self.B]) # confidence
        boxes = np.reshape(output[self.idx2:], [self.S, self.S, self.B, 4]) # boxes -> (x, y, w, h)

        # convert the x, y to the coordinates relative to the top left point of the image
        boxes[:, :, :, 0] += self.x_offset
        boxes[:, :, :, 1] += self.y_offset
        boxes[:, :, :, :2] /= self.S

        # the predictions of w, h are the square root
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # multiply the width and height of image
        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        # class-specific confidence scores [S, S, B, C]
        scores = np.expand_dims(confs, -1) * np.expand_dims(class_probs, 2)
        # (7,7,2,20) = (7,7,2,1) * (7,7,1,20)

        scores = np.reshape(scores, [-1, self.C]) # [S*S*B, C]
        boxes = np.reshape(boxes, [-1, 4])        # [S*S*B, 4]

        # filter the boxes when score < threhold
        scores[scores < self.threshold] = 0.0

        # non max suppression
        self.non_max_suppression(scores, boxes) # 非极大值抑制

        # report the boxes
        predict_boxes = [] # (class, x, y, w, h, score)
        max_idxs = np.argmax(scores, axis=1)
        for i in range(len(scores)):
            max_idx = max_idxs[i]
            if scores[i, max_idx] > 0.0: 
                predict_boxes.append((self.classes[max_idx], boxes[i, 0], boxes[i, 1],
                                      boxes[i, 2], boxes[i, 3], scores[i, max_idx]))
                                        # (类别, x, y, w, h, 可能性)
        return predict_boxes


    def non_max_suppression(self, scores, boxes): # 非极大值抑制
        # for each class
        for c in range(self.C):
            sorted_idxs = np.argsort(scores[:, c]) # 从大到小排序
            last = len(sorted_idxs) - 1
            while last > 0:
                if scores[sorted_idxs[last], c] < 1e-6: # 这个框不可能有这个类的东西
                    break
                for i in range(last): # 这个框可能有这个类的东西，检查另一个框们
                    if scores[sorted_idxs[i], c] < 1e-6: # 另一个框没有
                        continue
                    # 另一个框可能有，若交互比大于阈值（重叠度太高），则将另一个框置零
                    if self.iou(boxes[sorted_idxs[i]], boxes[sorted_idxs[last]]) > self.iou_threshold:
                        scores[sorted_idxs[i], c] = 0.0
                last -= 1


    def iou(self, box1, box2): # 计算这两个边框的交互比
        inter_w = np.minimum(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
                  np.maximum(box1[0]-0.5*box2[2], box2[0]-0.5*box2[2])
        inter_h = np.minimum(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
                  np.maximum(box1[1]-0.5*box2[3], box2[1]-0.5*box2[3])
        if inter_h < 0 or inter_w < 0:
            inter = 0
        else:
            inter = inter_w * inter_h
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return inter / union


    def show_results(self, img, results):
        #  draw boxes
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            print("   class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (results[i][0],
                            x, y, w, h, results[i][-1]))

            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.imshow('YOLO_small detection', img)
        cv2.waitKey(1)


    def train(self, images, targets):
        num = images.shape[0] # 训练样本数
        max_epoch = 1000
        loss = 0

        for epoch in range(max_epoch):
            for i in range(num):
                # 因为还不是很熟tensorflow所以底下这四行是瞎编的，反正就是反向传播让loss沿梯度下降
                output = self.sess.run(self.net_output, feed_dict={self.img_input: images[i]})[0]
                loss = get_loss(output, targets[i])
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate) # 优化器
                opt.minimize(loss)
            print('epoch:%f, loss:%f' % (epoch, loss))

        tf.save_weights('myWeights.ckpt') # 瞎编的函数，反正就是保存训练好的权重


    def get_loss(self, output, target): # 一张图片训练后与真实情况的损失 # output:7*7*30, target:7*7*25
        class_probs = np.reshape(output[:self.idx1], [self.S * self.S, self.C]) # class prediction
        confs = np.reshape(output[self.idx1:self.idx2], [self.S * self.S, self.B]) # confidence
        boxes = np.reshape(output[self.idx2:], [self.S * self.S, self.B, 4]) # boxes -> (x, y, w, h)
        boxes[:, :, 0] += self.x_offset
        boxes[:, :, 1] += self.y_offset
        boxes[:, :, :2] /= self.S
        boxes[:, :, 2:] = np.square(boxes[:, :, 2:])
        boxes[:, :, 0] *= img_w
        boxes[:, :, 1] *= img_h
        boxes[:, :, 2] *= img_w
        boxes[:, :, 3] *= img_h
        
        real_class_probs = np.reshape(target[:self.idx1], [self.S * self.S, self.C]) # class prediction
        real_confs = np.reshape(target[self.idx1:self.idx1 + 1], [self.S * self.S, 1]) # confidence
        real_boxes = np.reshape(target[self.self.idx1 + 1:], [self.S * self.S, 4]) # boxes -> (x, y, w, h)
        real_boxes[:, 0] += self.x_offset
        real_boxes[:, 1] += self.y_offset
        real_boxes[:, :2] /= self.S
        real_boxes[:, 2:] = np.square(real_boxes[:, 2:])
        real_boxes[:, 0] *= img_w
        real_boxes[:, 1] *= img_h
        real_boxes[:, 2] *= img_w
        real_boxes[:, 3] *= img_h
        
        loss = 0
        for i in range(self.S * self.S):
            if(real_class_probs[i].any()): # 这个网格真的有东西
                j = np.argmax([self.iou(boxes[i,j], real_boxes[i]) for j in range(self.B)]) # 只选B个框中与真框重合度最大的框
                loss += self.lambda_coord *\
                        ((boxes[i,j,0] - real_boxes[i,0])**2 + (boxes[i,j,1] - real_boxes[i,1])**2)
                loss += self.lambda_coord *\
                        ((np.sqrt(boxes[i,2]) - np.sqrt(real_boxes[i,2]))**2 + (np.sqrt(boxes[i,3]) - np.sqrt(real_boxes[i,3]))**2)
                loss += self.lambda_obj *\
                        (confs[i,j] - real_confs[i,0])**2
                for c in range(self.C):
                    loss += self.lambda_class *\
                            (class_probs[i,c] - real_class_probs[i,c])**2
            else: # 这个网格内没东西
                j = np.argmax([confs[i,j] for j in range(self.B)]) # 只选B个框中置信度最大的
                loss += self.lambda_noobj *\
                        (confs[i,j] - real_confs[i,0])**2

        return loss

                    
if __name__ == "__main__":
    yolo_net = Yolo()
    yolo_net.detect("./test_imgs/person.jpg", "./weights/YOLO_small.ckpt")
    '''
    images, targets = load_train_data()
    yolo_net.train(images, targets)
    '''
