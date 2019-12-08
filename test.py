import os

import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
a=tf.constant(np.array([1 , 2, 3]))
b=tf.constant(np.array([4,5,6]))
c1=tf.concat([a,b],axis=0)


sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c1))


