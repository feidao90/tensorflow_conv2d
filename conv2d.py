import tensorflow as tf
import numpy as  np

# import input_data

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# sess = tf.InteractiveSession()

sess = tf.Session()
# print(sess.run(tf.cross([2.,0.,0.],[1.,1.,0.])))

print(sess.run(tf.exp(-1.)))
print(sess.run(tf.exp(4.)))

x_vals = np.array([1.,3.,5.,7.,9.])
m_constant = tf.constant(3.)
x_data = tf.placeholder(tf.float32)
my_pro = tf.multiply(x_data,m_constant)

for x_val in  x_vals:
    print(sess.run(my_pro,feed_dict={x_data:x_val}))

my_array = np.array([[1.,3.,7.,9.],[-2.,0.,2.,4.,6.],[-6.,-3.,0.,3.,6.]])
x_vals = np.array([my_array,my_array + 1])

print(x_vals)

x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())


y = tf.nn.softmax(tf.matmul(x,w)+b)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

def custom_polynomial(value):
    return (tf.subtract(3 * tf.square(value),value) + 10)

print(sess.run(custom_polynomial(11)))
print(sess.run(tf.nn.relu([-3.,3.,10.])))   #max(0,x)

print(sess.run(tf.nn.relu6([-3.,3.,10.]))) #min(max(0,x),6)