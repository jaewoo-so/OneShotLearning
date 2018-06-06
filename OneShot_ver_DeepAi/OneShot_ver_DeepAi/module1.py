import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


res = [ tf.random_normal( [2,3] , 10.0 ,seed = 1) , tf.random_normal( [2,3] , 10.0 , seed = 2) ]
t1 , t2 = res[0] , res[1]
temp = t1 - t2
temp2 = tf.reduce_sum(temp)

sess = tf.Session(config = config)
print(sess.run([temp2]))
print(sess.run([temp2]))
print(sess.run([temp2]))
print(sess.run([temp2]))

print()
sess.close()


res = [ tf.random_normal( [2,3] , 10.0 ,seed = 1) , tf.random_normal( [2,3] , 10.0 , seed = 2) ]
t1 = tf.constant( res[0].eval(session = sess) )
t2 = tf.constant( res[1].eval(session = sess) )

temp = t1 - t2
temp2 = tf.reduce_sum(temp)

sess = tf.Session(config = config)
print(sess.run([temp2]))
print(sess.run([temp2]))
print(sess.run([temp2]))
print(sess.run([temp2]))

print()
sess.close()