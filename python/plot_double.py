'''This script is to plot the results of the actor critic for the double pendulum'''


from actorcritic import *
from pinocchio.utils import rand


# Sample a collection of points as [ A*rand(2) + B*fix(n-2) ] with A in R^nx2 and B in R^nx(n-2)
NBP = 10000
A = np.matrix([[ 0.,0.,1.,0.],[0.,0.,0.,1.]]).T
B = np.matrix([[ 1.,0.,0.,0.],[0.,1.,0.,0.]]).T
fix = [ np.array([0.,0.]),
        np.array([1.,0.]),
        np.array([2.,0.]),
        np.array([3.,0.]) ]
Xs = []
for f in fix:
    v = rand([NBP,2])*np.diag([2*np.pi,16])+np.matrix([-np.pi,-8])
    Xs.append( v*A.T+ f*B.T )

#plt.ion()
plt.figure(figsize=(16, 12), dpi=80)

for i in range(1,21):
    episode = i*3000-1
    RESTORE = "netvalues/double/actorcritic_double.%d" % episode
    print "*** Restore %i net weights from " %i,RESTORE
    tf.train.Saver().restore(sess, RESTORE)
    plt.clf()

    for iX,X in enumerate(Xs):
        U = sess.run(policy.policy, feed_dict={ policy.x: env.obs(X.T).T })
        Q = sess.run(qvalue.qvalue, feed_dict={ qvalue.x: env.obs(X.T).T,
                                                qvalue.u: U })
        # Scatter plot of policy/value funciton sampling (in file)
        nU = U.shape[1]
        for iU in range(nU):
            plt.subplot(len(Xs),nU+1,iU+iX*(nU+1)+1)
            print      (len(Xs),nU+1,iU+iX*(nU+1)+1)
            plt.scatter((X*A[:,0]).flat,(X*A[:,1]).flat,c=U[:,iU],s=50,linewidths=0,alpha=.8,vmin=-env.umax,vmax=env.umax)
            plt.xlabel('X = *'+str(A[:,0].T))
            plt.ylabel('X = *'+str(A[:,1].T))
            plt.colorbar()

        plt.title('offset = '+str(X[0,:]*B*B.T))
        plt.subplot(len(Xs),nU+1,nU+1+iX*(nU+1))
        print      (len(Xs),nU+1,nU+1+iX*(nU+1),"V")

            #plt.subplot(1,2,2)
        plt.scatter((X*A[:,0]).flat,(X*A[:,1]).flat,c=Q[:],s=50,linewidths=0,alpha=.8)
        plt.colorbar()
        #plt.savefig('figs/actorcritic_%04d.png' % episode)

    plt.savefig('figs/double_%04d.png' % episode)
    #break


