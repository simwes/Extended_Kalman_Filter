from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import numpy as np
from math import sqrt
from numpy.random import randn
import math
import kf_book.ekf_internal as ekf_internal
import matplotlib.pyplot as plt


from math import sqrt
def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """

    horiz_dist = x[0]
    altitude   = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return array ([[horiz_dist/denom, 0.,  altitude/denom, 0.]])

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    
    return (x[0]**2 + x[2]**2) ** 0.5

class RadarSim:
    """ Simulates the radar signal returns from an object
    flying at a constant altityude and velocity in 1D. 
    """
    
    def __init__(self, dt, pos, velX, velY, alt):
        self.pos  = pos
        self.velX = velX
        self.velY = velY
        self.alt  = alt
        self.dt   = dt
        
    def get_range(self):
        """ Returns slant range to the object. Call once 
        for each new measurement at dt time from last call.
        """
        
        # add some process noise to the system
        self.velX = self.velX  #+ .1*randn()
        self.velY = self.velY  #+ .1*randn()
        self.alt  = self.alt   +  self.velY*self.dt
        self.pos  = self.pos   +  self.velX*self.dt
    
        # add measurement noise
        err = self.pos * 0.05*randn()
        slant_dist = math.sqrt(self.pos**2 + self.alt**2)
        
        return slant_dist # + err


dt = 0.05
rk = ExtendedKalmanFilter( dim_x=4, dim_z=1 )
radar = RadarSim( dt, pos=10., velX=0.5, velY=1., alt=10. )

# make an imperfect starting guess
rk.x = array( [ radar.pos, radar.velX, radar.alt, radar.velY ] )

rk.F = eye(4) + array([[0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]]) * dt

range_std = .5 # meters
rk.R = np.diag([range_std**2])
rk.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.01)
rk.Q[2,2] = 0.1
rk.P *= 50

xs, track = [], []
for i in range(int(20/dt)):
    z = radar.get_range()
    track.append((radar.pos, radar.velX, radar.alt, radar.velY ))
    
    rk.update( array([z]), HJacobian_at, hx )
    xs.append( rk.x )
    rk.predict()

xs = asarray(xs)
track = asarray(track)
time = np.arange(0, len(xs)*dt, dt)
#ekf_internal.plot_radar(xs, track, time)

print(xs.shape, track.shape)

plt.figure()
plt.plot( time, track[:,0],'or', label='X filter' )
plt.plot( time, xs[:,0],  'b', label='X track' )

plt.plot( time, track[:,2], 'og', label='Y filter' )
plt.plot( time, xs[:,2],  'y',  label='Y track' )

plt.legend()

plt.figure()
plt.plot( time, xs[:,1], label='Ux' )
plt.plot( time, xs[:,3], label='Uy' )
plt.legend()


plt.show()