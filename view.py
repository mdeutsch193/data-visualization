# Martin Deutsch
# Project 8
# CS 251
# Spring 2017

import numpy as np
import math

# A class to hold viewing parameters and build view matrix
class View:
    
    # initialize view paramters
    def __init__(self):
        self.vrp = np.matrix([])
        self.vpn = np.matrix([])
        self.vup = np.matrix([])
        self.u = np.matrix([])
        self.extent = []
        self.screen = []
        self.offset = []
        
        self.reset()
    
    # give view parameters default values
    def reset(self):
        self.vrp = np.matrix([0.5, 0.5, 1])
        self.vpn = np.matrix([0, 0, -1])
        self.vup = np.matrix([0, 1, 0])
        self.u = np.matrix([-1, 0, 0])
        self.extent = [1, 1, 1]
        self.screen = [400, 400]
        self.offset = [20, 20]
    
    # create view transformation matrix
    def build(self):
        # initialize view transformation matrix
        vtm = np.identity( 4, float )
        # translate VRP to origin
        t1 = np.matrix( [[ 1, 0, 0, -self.vrp[0, 0] ],
                         [ 0, 1, 0, -self.vrp[0, 1] ],
                         [ 0, 0, 1, -self.vrp[0, 2] ],
                         [ 0, 0, 0, 1 ] ] )

        vtm = t1 * vtm
        # Calculate orthonormal axes
        tu = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn
        tu = self.normalize(tu)
        tvup = self.normalize(tvup)
        tvpn = self.normalize(tvpn)
        self.u = tu
        self.vup = tvup
        self.vpn = tvpn
        # align the axes
        r1 = np.matrix( [[ tu[0, 0], tu[0, 1], tu[0, 2], 0.0 ],
                         [ tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0 ],
                         [ tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0 ],
                         [ 0.0, 0.0, 0.0, 1.0 ] ] )
        vtm = r1 * vtm
        # translate lower left of view space to origin
        t2 = np.matrix( [[ 1.0, 0.0, 0.0, 0.5*self.extent[0] ],
                         [ 0.0, 1.0, 0.0, 0.5*self.extent[1] ],
                         [ 0.0, 0.0, 1.0, 0.0 ],
                         [ 0.0, 0.0, 0.0, 1.0 ] ] )
        vtm = t2 * vtm
        # scale the screen
        s1 = np.matrix( [[ -self.screen[0]/self.extent[0], 0.0, 0.0, 0.0 ],
                         [ 0.0, -self.screen[1]/self.extent[1], 0.0, 0.0 ],
                         [ 0.0, 0.0, 1.0/self.extent[2], 0.0 ],
                         [ 0.0, 0.0, 0.0, 1.0 ] ] )
        vtm = s1 * vtm
        # translate lower left of view space to origin and add buffer
        t3 = np.matrix( [[ 1, 0, 0, self.screen[0]+self.offset[0] ],
                         [ 0, 1, 0, self.screen[1]+self.offset[1] ],
                         [ 0, 0, 1, 0 ],
                         [ 0, 0, 0, 1 ] ] )
        vtm = t3 * vtm
        return vtm
        
    # normalize given vector
    def normalize(self, v):
        length = math.sqrt( v[0, 0]*v[0, 0] + v[0, 1]*v[0,1] + v[0,2]*v[0,2] )
        return v / length
       
    # create new View object with same fields as current View object
    def clone(self):
        newView = View()
        newView.vrp = self.vrp
        newView.vpn = self.vpn
        newView.vup = self.vup
        newView.u = self.u
        newView.extent = self.extent
        newView.screen = self.screen
        newView.offset = self.offset
        return newView
    
    # rotate about the center of the view volume
    def rotateVRC(self, VUProtation, Urotation):
        t1 = np.matrix( [[ 1, 0, 0, -(self.vrp[0,0] + self.vpn[0,0] * self.extent[2] * 0.5) ],
                         [ 0, 1, 0, -(self.vrp[0,1]+ self.vpn[0,1] * self.extent[2] * 0.5) ],
                         [ 0, 0, 1, -(self.vrp[0,2]+ self.vpn[0,2] * self.extent[2] * 0.5) ],
                         [ 0, 0, 0, 1 ] ] )
        Rxyz = np.matrix( [[ self.u[0,0], self.u[0,1], self.u[0,2], 0.0 ],
                           [ self.vup[0,0], self.vup[0,1], self.vup[0,2], 0.0 ],
                           [ self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0.0 ],
                           [ 0.0, 0.0, 0.0, 1.0 ] ] )
        r1 = np.matrix( [[ math.cos(VUProtation), 0, math.sin(VUProtation), 0 ],
                         [ 0, 1, 0, 0 ],
                         [ -math.sin(VUProtation), 0, math.cos(VUProtation), 0 ],
                         [ 0, 0, 0, 1 ] ] )
        r2 = np.matrix( [[ 1, 0, 0, 0 ],
                         [ 0, math.cos(Urotation), -math.sin(Urotation), 0 ],
                         [ 0, math.sin(Urotation), math.cos(Urotation), 0 ],
                         [ 0, 0, 0, 1 ] ] )
        t2 = np.matrix( [[ 1, 0, 0, self.vrp[0,0] + self.vpn[0,0] * self.extent[2] * 0.5 ],
                         [ 0, 1, 0, self.vrp[0,1]+ self.vpn[0,1] * self.extent[2] * 0.5 ],
                         [ 0, 0, 1, self.vrp[0,2]+ self.vpn[0,2] * self.extent[2] * 0.5 ],
                         [ 0, 0, 0, 1 ] ] )
        tvrc = np.matrix( [[ self.vrp[0,0],self.vrp[0,1], self.vrp[0,2], 1 ],
                           [ self.u[0,0], self.u[0,1], self.u[0,2], 0 ],
                           [ self.vup[0,0], self.vup[0,1], self.vup[0,2], 0 ],
                           [ self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0 ] ] )
        tvrc = (t2 * Rxyz.T * r2 * r1 * Rxyz * t1 * tvrc.T).T
        self.vrp = tvrc[0, :3]
        self.u = self.normalize(tvrc[1, :3])
        self.vup = self.normalize(tvrc[2, :3])
        self.vpn = self.normalize(tvrc[3, :3])