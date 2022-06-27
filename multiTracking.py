from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import numpy as np
from math import sqrt
from numpy.random import randn
import math
import kf_book.ekf_internal as ekf_internal
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
#from triangulation import *
import cv2,csv
from sklearn.neighbors import NearestNeighbors
#from preProcessing_chunks import preProcessingChunk
from preProcessing import preProcessing
#from displayPoint import video
from colorLine import *
from scipy import spatial



fish = 6

r2 = 0.01#0.0275 # fish 3 =0.02905 # fish 2 =0.025# fish 1 = 0.03

nFish  = 10
frac   = .305#.182
nFrame = 9600 #95

dt   = 0.017 #0.0172

rVal = 0.05   #  more weight on the model
qVar = 0.01
qVal = 0.5 #8    #0.005 # more weight on measurement
pVal = 0.1

rValZ = 0.01   #  more weight on the model
qVarZ = 0.01

qValZ = 0.5#8    #0.005 # more weight on measurement
pValZ = 0.1


nDim = 2 # problem dimensions 

xScal = 640#*0.264
yScal = 360#*0.264 

crossPoint = 47

# ---------------------- From post-processing chunk -------------------
#pointsNz, features, missedData = preProcessingChunk( frac=frac, nFish=nFish, nFrame=nFrame )

# ---------------------- From post-processing chunk -------------------
pointsNz, features, features02, missedData = preProcessing( frac=frac, nFish=nFish, nFrame=nFrame, dt=dt )

print('')
print( pointsNz )
print('')
print(pointsNz.shape)



def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """
    horiz_dist = x[0]
    #altitude   = x[2]
    #denom = sqrt( horiz_dist**2  )
    return array ( [[1, 0.]] )

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """   
    return x[0]

class measurements:
    """ Simulates the radar signal returns from an object
    flying at a constant altityude and velocity in 1D. 
    """    
    def __init__(self, dt, pos, vel, iTime, iCamera,  jDir):
        self.pos   = pos 
        self.vel   = vel
        self.jDir  = jDir
        self.dt    = dt
        self.iTime = iTime
        self.iCamera = iCamera
        
    def get_rangeX(self, pos):
        """ Returns slant range to the object. Call once 
        for each new measurement at dt time from last call.
        """        
        # add some process noise to the system
        #self.iTime = self.iTime + 1


        if (self.jDir==0 and self.iCamera==0):
            self.pos  = self.pos#xmL[self.iTime]
            self.vel = self.vel


    def get_rangeY(self, pos):

        if (self.jDir==1 and self.iCamera==0):
            self.pos  = self.pos#ymL[self.iTime]
            self.vel = self.vel

                    
        slant_dist = math.sqrt(self.pos**2 )
        
        return slant_dist 


iF = fish-1

start = 2*fish-1
end   = 2*fish

rkXl = np.zeros(nFish)

#print(points)    

skippedIndex = []

rkXl    = ExtendedKalmanFilter( dim_x=2, dim_z=1 )
rkYl    = ExtendedKalmanFilter( dim_x=2, dim_z=1 )
rkZl    = ExtendedKalmanFilter( dim_x=2, dim_z=1 )
rkDl    = ExtendedKalmanFilter( dim_x=2, dim_z=1 )
v0l     = np.zeros(4)
pos0l   = np.zeros(4)

ekfPosl = np.zeros( shape=(3, pointsNz[:,0].size-2) )
ekfVell = np.zeros( shape=(3, pointsNz[:,0].size-2) )


range_std = rVal #0.05 # meters
rkXl.R = np.diag([range_std**2])# measurement noise covariance
rkXl.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=qVar) # process noise covariance (e.g v*t)
rkXl.Q[1,1] = qVal#0.005
rkXl.P *= pVal#0.5 # state  covariance (e.g. uncertainty on the position x)
xsXl = []


range_std = rVal #0.05 # meters
rkYl.R = np.diag([range_std**2])# measurement noise covariance
rkYl.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=qVar) # process noise covariance (e.g v*t)
rkYl.Q[1,1] = qVal#0.005
rkYl.P *= pVal#0.5 # state  covariance (e.g. uncertainty on the position x)
xsYl = []

range_std = rValZ #0.05 # meters
rkZl.R = np.diag([range_std**2])# measurement noise covariance
rkZl.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=qVarZ) # process noise covariance (e.g v*t)
rkZl.Q[1,1] = qValZ#0.005
rkZl.P *= pValZ#0.5 # state  covariance (e.g. uncertainty on the position x)
xsZl = []

range_std = rValZ #0.05 # meters
rkDl.R = np.diag([range_std**2])# measurement noise covariance
rkDl.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=qVarZ) # process noise covariance (e.g v*t)
rkDl.Q[1,1] = qValZ#0.005
rkDl.P *= pValZ#0.5 # state  covariance (e.g. uncertainty on the position x)
xsDl = []



#tracks  = np.zeros( shape = (nFish, pointsNz[:,0].size, nDim + 1) ) 
tracks   =  np.zeros( shape = ( pointsNz[:,0].size, 2*nFish + 1) ) 
vels     =  np.zeros( shape = ( pointsNz[:,0].size, 2*nFish + 1) ) 

tracksF  =  np.zeros( shape = ( pointsNz[:,0].size, nFish + 1) ) 
velsF    =  np.zeros( shape = ( pointsNz[:,0].size, nFish + 1) )

tracksD  =  np.zeros( shape = ( pointsNz[:,0].size, nFish + 1) ) 
velsD    =  np.zeros( shape = ( pointsNz[:,0].size, nFish + 1) )

zXl      =  np.zeros( shape = ( pointsNz[:,0].size, nFish ) ) 
zYl      =  np.zeros( shape = ( pointsNz[:,0].size, nFish ) )
zZl      =  np.zeros( shape = ( pointsNz[:,0].size, nFish ) ) 
zDl      =  np.zeros( shape = ( pointsNz[:,0].size, nFish ) ) 

xMissDet, yMissDet, zMissDet,  dMissDet = [], [], [], []
rkXl_pred, rkYl_pred, rkZl_pred = [], [], []
rkXl_update, rkYl_update, rkZl_update = [], [], []
zX, zY, zZ, zD = [], [], [], []
knnX, knnY, knnZ = [], [], []

knnFrameX, knnFrameY = [], []


#print('pointsNz.shape', pointsNz.shape)

t = 0
for i in range(0, pointsNz[:,0].size):
    t = t + dt

    #print('Keep',pointsNz[i,0], pointsNz[i,1],pointsNz[i,2], pointsNz[i,3],pointsNz[i,4])
    print('')

    for iFish in range( start, end ):

        print('i:', i, 'time:', t, 'iFish:', iFish, 'Start:', start, 'end:', end)
                    
        pos0l[0] = pointsNz[i, iFish]   # xmL[0]
        pos0l[1] = pointsNz[i, iFish+1] # ymL[0]
        pos0l[2] = features[i, fish]

        if i == 0: #this goes before iFish loop to make it automatic

            v0l[0]  = .00005
            v0l[1]  = .00005
            v0l[2]  = .00005
            
            zXl[i,iF] = pointsNz[i, iFish]
            zYl[i,iF] = pointsNz[i, iFish+1]
            zZl[i,iF] = features[i, fish]

            print('Initialize position:', zXl[i,iF]*xScal, zYl[i,iF]*yScal, zZl[i,iF]*xScal,features[i, fish]*xScal )
            print('Initialize velocity:', v0l[0], v0l[1], v0l[2])
            
         #  ======================== X component =================
            radarX_l = measurements( dt, pos=pos0l[0], vel=v0l[0], iTime=0, iCamera=0, jDir=0 )
    
            rkXl.x = array( [ radarX_l.pos, radarX_l.vel ] )
            rkXl.F = eye(2) + array( [[0, 1],
                                      [0, 0] ] ) * dt
            
         # ========================= Y component =================
            radarY_l = measurements( dt, pos=pos0l[1], vel=v0l[1], iTime=0, iCamera=0, jDir=1 )

            rkYl.x = array( [ radarY_l.pos, radarY_l.vel ] )
            rkYl.F = eye(2) + array( [[0, 1],
                                      [0, 0] ] ) * dt
            
        # ========================== Z component =================
            radarZ_l = measurements( dt, pos=pos0l[2], vel=v0l[2], iTime=0, iCamera=0, jDir=1 )

            rkZl.x = array( [ radarZ_l.pos, radarZ_l.vel ] )
            rkZl.F = eye(2) + array( [[0, 1],
                                      [0, 0] ] ) * dt
        else:

            tracks[i, 0] = pointsNz[i, 0]

             #----- for X ------
            print('Observation position: ', zXl[i-1, iF]*xScal, zYl[i-1, iF]*yScal, zZl[i-1, iF]*xScal)

            rkXl.update( array([zXl[i-1, iF]]), HJacobian_at, hx )

            tracks[i, iFish] = rkXl.x[0]
            vels[i, iFish]   = rkXl.x[1]
    
            xsXl.append( rkXl.x )
            rkXl.predict()


            # #----- for Y ------
            rkYl.update( array([zYl[i-1, iF]]), HJacobian_at, hx )

            tracks[ i, iFish+1] = rkYl.x[0]
            vels[ i, iFish+1]   = rkYl.x[1]
                        
            xsYl.append( rkYl.x )
            rkYl.predict()


            # #----- for Z ------
            tracksF[i, 0] = features[i, 0] # this is the time line
            rkZl.update( array([zZl[i-1, iF]]), HJacobian_at, hx )

            print(iFish, tracksF.shape)
            tracksF[ i, fish] = rkZl.x[0]
            velsF[ i, fish]   = rkZl.x[1]
                        
            xsZl.append( rkZl.x )
            rkZl.predict()
            
            #-----------------------------------------------------
            #---------------- Data Association -------------------

            # 3D knn here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree

            # ------------ Select points within a circle ------------

            xIndex, yIndex, fIndex = [], [], []

            for kFish in range( 1, nFish + 1 ):

                dist = math.sqrt( (pointsNz[i, 2*kFish-1] - rkXl.x[0])**2 +  \
                                    (pointsNz[i, 2*kFish] - rkYl.x[0])**2  +  \
                                    (features[i, kFish] - rkZl.x[0])**2 )

                print('dist:', dist)
                if r2 >= dist:
                    
                    xIndex.append(2*kFish-1)
                    yIndex.append(2*kFish)
                    fIndex.append(kFish)
                    

            # ------------- New Association ---------
            if len(xIndex)>0:        
            
                dataFrame = list( zip( pointsNz[i, xIndex[:]], pointsNz[i, yIndex[:]], features[i, fIndex[:]] ) )

                tree = spatial.KDTree(dataFrame)

                point = [rkXl.x[0], rkYl.x[0], rkZl.x[0]]

                distance, index = tree.query( np.array( [ point ]) )


                # # print( 'points:', len(xIndex) )             
                print( 'prediction position:', rkXl.x[0]*xScal, rkYl.x[0]*yScal, rkZl.x[0]*xScal )
                print( 'prediction velocity:', rkXl.x[1]*xScal, rkYl.x[1]*yScal, rkZl.x[1]*xScal )

                #  ------------------------ New association --------------------                
                zXl[i, iF] = dataFrame[index[0]][0]
                zYl[i, iF] = dataFrame[index[0]][1]
                zZl[i, iF] = dataFrame[index[0]][2]

                if len(xIndex)>1: 

                    with open('knnFrames/knn%d_f%d.csv' % ( fish, i ), 'w') as f:
                           writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
                           writer.writerow( [ 'X',  'Y', 'BB', 'xIndex', 'yIndex' ] )
                           writer.writerows( zip( pointsNz[i, xIndex[:]]*xScal, pointsNz[i, yIndex[:]]*yScal, features[i, fIndex[:]]*xScal, xIndex[:], yIndex[:]  ) )
                           writer.writerow( [ dataFrame[index[0]][0]*xScal, dataFrame[index[0]][1]*yScal, dataFrame[index[0]][2]*xScal ] )

                    with open('knnFrames/est%d_f%d.csv' % ( fish, i ), 'w') as f:
                           writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
                           writer.writerow( [ 'X',  'Y' ] )                           
                           writer.writerow( [ rkXl.x[0]*xScal, rkYl.x[0]*yScal, rkZl.x[0]*xScal ] )
                          

            else:

              # --------- Missed detection: it uses linear model to interpolate ------------

              zXl[i, iF] = rkXl.x[0]
              zYl[i, iF] = rkYl.x[0]
              zZl[i, iF] = rkZl.x[0]

              xMissDet.append(rkXl.x[0]*xScal)
              yMissDet.append(rkYl.x[0]*yScal)
              zMissDet.append(rkZl.x[0]*xScal)


with open('tFish%d.csv' % fish, 'w') as f:
       writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
       writer.writerow( [ 'frame',  'X', 'Y' ] )
       writer.writerows( zip( tracks[  1:, 0], tracks[  1:, start]*xScal, tracks[  1:, end]*yScal  ))




#   ========================================================================
print('')
print('tracks',tracks.shape)
print('Missed detection on a single fish', len(xMissDet))
print('prediction position:', rkXl.x[0], rkYl.x[0] )
print('Overall miss detection:', missedData,'%',len(xMissDet))



#   ====================== Make video with tracking dots =====================

xsXl = asarray(xsXl)
#print( ekfPosl.shape , xsXl.shape )

xsYl = asarray(xsYl)

timel = np.arange(0, len(xsXl)*dt, dt)
# timer = np.arange(0, len(xsXr)*dt, dt)

#==============================================================
# --------------------------- Plot ---------------------------
print('')
print('---------------------- Plot --------------------')
#print(timel.size, xmL[:-2].size)
# # -------======================= Left Camera =======================-----------
plt.figure(1)
plt.subplot(2, 1, 1)
#plt.scatter(timel, xmL[:-2], s=20, facecolors='none', edgecolors='r', label='X' )
plt.plot( timel, tracks[  1:, start]*xScal,'-o', color='b',  label='X filter' )
plt.legend()

#plt.scatter(timel, ymL[:-2], s=20, facecolors='none', edgecolors='b', label='Y' )
plt.plot( timel, tracks[  1:, end]*yScal,'-o' ,color='r', label='Y filter' )
plt.xlabel('Time[s]')
plt.ylabel('Trajectory components [px]')
plt.title('Left Camera')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot( timel, vels[  1:, start]*xScal, '-o',color='b', label='Ux' )
plt.plot( timel, vels[  1:, end]*yScal, '-o',color='r', label='Uy' )
plt.xlabel('Time[s]')
plt.ylabel('Trajectory components [px/s]')
plt.legend()



plt.figure(2)
for iFish in range(1, nFish+1):
    plt.scatter( pointsNz[:,2*iFish-1]*xScal, pointsNz[:,2*iFish]*yScal, s=35, facecolors='none', edgecolors='b' )
# plt.scatter( pointsNz[:,3]*xScal, pointsNz[:,4]*yScal, s=35, facecolors='none', edgecolors='r', label='Fish 2')
# plt.scatter( pointsNz[:,5]*xScal, pointsNz[:,6]*yScal, s=35, facecolors='none', edgecolors='g', label='Fish 3')


#plt.plot( tracks[  1:, start]*xScal, tracks[ 1:, end]*yScal,'-*', color='g', label='Filter' )

colorline( tracks[  1:, start]*xScal, tracks[ 1:, end]*yScal,cmap=plt.get_cmap('hot'), )
plt.plot( xMissDet, yMissDet,'+', color='r', label='Missed' )


# for i in range(0, len(rkXl_pred)):
#   plt.gca().add_patch(plt.Circle((rkXl_pred[i]*xScal, rkYl_pred[i]*yScal), r2*xScal, color='g', fill=False))
#   plt.plot(rkXl_pred[i]*xScal, rkYl_pred[i]*yScal,'+', color='g')
#   plt.text(rkXl_pred[i]*xScal + rkXl_pred[i]*xScal*0.0001, rkYl_pred[i]*yScal + 0.01, '%d' % (i+45), fontsize=9)

#   plt.plot(rkXl_update[i]*xScal, rkYl_update[i]*yScal,'*', color='r')
#   plt.text(rkXl_update[i]*xScal + rkXl_update[i]*xScal*0.0001, rkYl_update[i]*yScal + 0.01, '%d' % (i+44), fontsize=9)

# for i in range(crossPoint - 3, crossPoint + 6):
#   plt.plot( zXl[i, iF]*xScal, zYl[i, iF]*yScal,'+', color='yellow' )
#   plt.text(zXl[i, iF]*xScal + zXl[i, iF]*xScal*0.0001, zYl[i, iF]*yScal + 0.01, '%d' % i, fontsize=9)


plt.ylabel('Y[mm]')
plt.xlabel('X[mm]')
sec=nFrame*frac/60
plt.title('Fish %d' % fish + ' frac %f' % frac +', %fs ' % sec)

plt.legend()
plt.xlim(80, 540)
plt.ylim(15, 360)

# plt.xlim(428*0.264, 480*0.264)
# plt.ylim(300*0.264, 340*0.264)
plt.gca().invert_yaxis()


#print(pointsNz.shape, features.shape)

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')


# for iFish in range(1, nFish+1):
#   ax.scatter( pointsNz[:,2*iFish-1]*xScal, pointsNz[:,2*iFish]*yScal, features[:,iFish]*yScal, alpha=0.5, color='b' )

# ax.plot3D( tracks[  1:, start]*xScal, tracks[ 1:, end]*yScal, tracksF[ 1:, fish]*yScal, linewidth=3, color='r' )
# # #ax.plot( xMissDet, yMissDet,'+', color='r', label='Missed' )

# #plt.xlim(60, 140)
# # plt.ylim(300*0.264, 340*0.264)
# ax.set_xlabel('X[mm]')
# ax.set_ylabel('Y[mm]')
# ax.set_zlabel('BB length[mm]')

plt.show()
