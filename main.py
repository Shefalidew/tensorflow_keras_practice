import tensorflow.compat.v1 as tf
tf.disable_v2_behaviour()

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype = tf.float32,shape = (None), name= 'x')
    w = tf.Variable(2.0,name='weight')
    b = tf.Variable(0.7, name ='bias')
    z= w*x +b

    init = tf.global_variables_initializer()

with tf.Session(graph= g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x:[1.,2.,3.]}))
    for t in [1.0 , 0.6, -1.8]:
        print('x = %4.1f --> z=%4.1f'%(t, sess.run(z,feed_dict={x:t})))