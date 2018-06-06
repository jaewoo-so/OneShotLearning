import functools
import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess =tf.Session(config = config)
tf.set_random_seed(1)

def Check_Eval_changeRnadom():
    y_pred1 = (tf.random_normal([3, 12], mean=6, stddev=0.1, seed = 1),
                  tf.random_normal([3, 12], mean=1, stddev=1, seed = 1),
                  tf.random_normal([3, 12], mean=3, stddev=4, seed = 1))
        
    y_pred2 = (tf.random_normal([3, 12], mean=6, stddev=0.1, seed = 1),
                  tf.random_normal([3, 12], mean=1, stddev=1, seed = 1),
                  tf.random_normal([3, 12], mean=3, stddev=4, seed = 1))
        
    y_pred3 = (tf.random_normal([3, 12], mean=6, stddev=0.1, seed = 1),
                  tf.random_normal([3, 12], mean=1, stddev=1, seed = 1),
                  tf.random_normal([3, 12], mean=3, stddev=4, seed = 1))
    
    
    anchor1, positive1, negative1 =  y_pred1[0], y_pred1[1], y_pred1[2]
    anchor2, positive2, negative2 =  y_pred2[0], y_pred2[1], y_pred2[2]
    anchor3, positive3, negative3 =  y_pred3[0], y_pred3[1], y_pred3[2]
    
    alpha = 0.2
    
    pos_dist1 = tf.reduce_sum( tf.square( anchor1 - positive1) , axis = 1)    
    neg_dist1 = tf.reduce_sum(tf.square(anchor1 - negative1) , axis = 1)
    basic_loss1 = tf.add( tf.subtract(pos_dist1 , neg_dist1) , alpha)
    loss1 = tf.reduce_mean( tf.maximum(basic_loss1 , 0.0) , axis = 0)
    
    pos_dist2 = tf.square(anchor2 - positive2)
    neg_dist2 = tf.square(anchor2 - negative2)
    basic_loss2 = tf.reduce_sum(pos_dist2-neg_dist2)+alpha
    loss2 = tf.reduce_sum(tf.maximum(basic_loss2,0.))
            
    pos_dist3 = tf.reduce_sum( tf.square(anchor3 - positive3))
    neg_dist3 = tf.reduce_sum( tf.square(anchor3 - negative3))
    basic_loss3 = tf.add(tf.subtract(pos_dist3, neg_dist3), alpha)
    loss3 = tf.maximum(tf.reduce_mean(basic_loss3), 0.0)
    
    l1 , l2 , l3 = sess.run([loss1 , loss2 , loss3])
    print("l1 : {} , l2 = {} , l3 = {}".format(l1,l2,l3))
    l1 , l2 , l3 = sess.run([loss1 , loss2 , loss3])
    print("l1 : {} , l2 = {} , l3 = {}".format(l1,l2,l3))

def Check_fixRnadom():
    y_pred1 = (tf.random_normal([3, 12], mean=6, stddev=0.1, seed = 1),
               tf.random_normal([3, 12], mean=1, stddev=1, seed = 1),
               tf.random_normal([3, 12], mean=3, stddev=4, seed = 1))
    alpha = 0.2
    evallist = list( map(lambda x : x.eval(session = sess) , y_pred1) )

    anchor   = tf.constant(evallist[0])
    positive = tf.constant(evallist[1])
    negative = tf.constant(evallist[2])

    anchor1   = anchor  
    positive1 = positive
    negative1 = negative

    anchor2   = anchor  
    positive2 = positive
    negative2 = negative

    anchor3   = anchor  
    positive3 = positive
    negative3 = negative

    pos_dist1 = tf.reduce_sum( tf.square( anchor1 - positive1) , axis = 1) 
    neg_dist1 = tf.reduce_sum(tf.square(anchor1 - negative1) , axis = 1)
    basic_loss1 = tf.add( tf.subtract(pos_dist1 , neg_dist1) , alpha)
    loss1 = tf.reduce_mean( tf.maximum(basic_loss1 , 0.0) , axis = 0)
    
    pos_dist2 = tf.square(anchor2 - positive2)
    neg_dist2 = tf.square(anchor2 - negative2)
    basic_loss2 = tf.reduce_sum(pos_dist2-neg_dist2)+alpha
    loss2 = tf.reduce_sum(tf.maximum(basic_loss2,0.))
            
    pos_dist3 = tf.reduce_sum( tf.square(anchor3 - positive3))
    neg_dist3 = tf.reduce_sum( tf.square(anchor3 - negative3))
    basic_loss3 = tf.add(tf.subtract(pos_dist3, neg_dist3), alpha)
    loss3 = tf.maximum(tf.reduce_mean(basic_loss3), 0.0)

    l1 , l2 , l3 = sess.run([loss1 , loss2 , loss3])
    print("l1 : {} , l2 = {} , l3 = {}".format(l1,l2,l3))
    l1 , l2 , l3 = sess.run([loss1 , loss2 , loss3])
    print("l1 : {} , l2 = {} , l3 = {}".format(l1,l2,l3))


def Check_TripletLoss():
    y_pred1 = (tf.random_normal([3, 12], mean=6, stddev=0.1, seed = 1),
               tf.random_normal([3, 12], mean=1, stddev=1, seed = 1),
               tf.random_normal([3, 12], mean=3, stddev=4, seed = 1))
    alpha = 0.2
    evallist = list( map(lambda x : x.eval(session = sess) , y_pred1) )

    anchor   = tf.constant(evallist[0])
    positive = tf.constant(evallist[1])
    negative = tf.constant(evallist[2])

    anchor1   = anchor  
    positive1 = positive
    negative1 = negative

    anchor2   = anchor  
    positive2 = positive
    negative2 = negative

    anchor3   = anchor  
    positive3 = positive
    negative3 = negative

    pos_dist1 = tf.reduce_sum( tf.square( anchor1 - positive1) , axis = 1) 
    neg_dist1 = tf.reduce_sum(tf.square(anchor1 - negative1) , axis = 1)
    substract = pos_dist1 - neg_dist1
    basic_loss1 = tf.add( tf.reduce_sum( tf.subtract(pos_dist1 , neg_dist1) ), alpha)
    loss1 =  tf.maximum(basic_loss1 , 0.0) 
    
    pos_dist2 = tf.square(anchor2 - positive2)
    neg_dist2 = tf.square(anchor2 - negative2)
    beforeSum2 = pos_dist2-neg_dist2
    basic_loss2 = tf.reduce_sum(pos_dist2-neg_dist2)+alpha
    loss2 = tf.maximum(basic_loss2,0.)
            
    pos_dist3 = tf.reduce_sum( tf.square(anchor3 - positive3))
    neg_dist3 = tf.reduce_sum( tf.square(anchor3 - negative3))
    basic_loss3 = tf.add(tf.subtract(pos_dist3, neg_dist3), alpha)
    loss3 = tf.maximum(basic_loss3, 0.0)

    l1 , l2 , l3 = sess.run([loss1 , loss2 , loss3])
    print("l1 : {} , l2 = {} , l3 = {}".format(l1,l2,l3))

    p1 , p2 , p3 = sess.run([pos_dist1 , pos_dist2 , pos_dist3])
    n1 , n2 , n3 = sess.run([neg_dist1 , neg_dist2 , neg_dist3])
    b1 , b2 , b3 = sess.run([basic_loss1 , basic_loss2 , basic_loss3])
    t1 , t2 = sess.run([ substract , beforeSum2 ])

    
    p2reduce = np.sum( p2 , axis = 1 )
    n2reduce = np.sum( n2 , axis = 1 )
    t2reduce = np.sum( t2 )
    print("t2 shape = {}".format(t2.shape))    

    print("p1 : {} , p2 = {} ".format(p1,p2reduce))
    print()

#Check_Eval_changeRnadom()
#Check_fixRnadom()
Check_TripletLoss() # 마지막에 모든 값을 Sum 한후 Max를 찾아야 했는데, 각 샘플당 Max값을 찾았다. .. 어쩌면 모든 값이 아니라 배치사이즈 만큼 일수도.. 
