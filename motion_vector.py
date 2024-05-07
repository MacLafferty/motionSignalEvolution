#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:45:16 2021

@author: davidmorris
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import circmean
#this is the default test video for the package
#opencv can be REALLY weird about relative file paths
#use full path to be safe
#cap = cv.VideoCapture("/Users/davidmorris/Biology/smile_and_wave/vtest.avi")
#let's try a habitat video!
cap = cv.VideoCapture("/Users/davidmorris/Biology/smile_and_wave/HADE_GLT_F61_S.MOV")
#another habitat option
#cap = cv.VideoCapture("/Users/davidmorris/Biology/smile_and_wave/HAAM_VIV_F97_N.MOV")

#cap = cv.VideoCapture("/Users/davidmorris/Biology/smile_and_wave/HADO_edited.mp4")
"""
#output matrix info to file if needed
#assume input is always gonna be hsv from the frame comparison
#3D matrix, where dimensions 1 and 2 are the pixel grid
#3rd index contains the calculated motion vectors, where 0th is angle and 2nd is magnitude
def crunchMotion(motionData,outputFileName):
    
    outputFile=open(outputFileName,"w")
    
    height=len(motionData)
    width=len(motionData[0])
    
    for row in range(0,height):
        for col in range(0,width):
            #grab angle for vector
            angle=motionData[row,col,0]
            #grab magnitude for vector
            mag=motionData[row,col,2]
            #tab separate vector info, then go to new line
            if mag > 0:
                outputFile.write(str(angle)+"\t"+str(mag)+"\n")
            
    
    outputFile.close()
"""

#pull a bunch of metadata about the video itself
#length and framerate useful for extracting set video length from middle
#might need dimensions for masking stuff
vid_frames= int(cap.get(cv.CAP_PROP_FRAME_COUNT))
vid_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
vid_fps    = cap.get(cv.CAP_PROP_FPS)
vid_length = vid_frames / vid_fps

#determine start and end frames
#select total length of video to analyze
analysis_length=15
analysis_frames=analysis_length * vid_fps
remainder_frames=vid_frames-analysis_frames
startFrame=round(0+remainder_frames/2)
#proper end frame code
endFrame=round(vid_frames-remainder_frames/2)
#fast video option
endFrame=startFrame + 90


#set the video capture object to first frame of interest
#first value is frame position ID
cap.set(1,startFrame)
#frame coomparison 81/82 has the greatest total motion values
#set frame here to specifically see lots of motion
#cap.set(1,81)
#this reads the next frame as our reference frame
ret, frame1 = cap.read()
#convert the frame to grey scale
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
#gaussian filter blurs image, accounts for sensor noise and is closer to spider visual system
#adjust this so we've got our data adjusted for our spiders
prvs = cv.GaussianBlur(prvs,(5,5),1)


#build matrix for motion vectors
#Hue/Saturation/Value matrix for displaying video frames later
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

#list of magnitudes (haha, community reference) to see how amounts of motion changes over time
popPop=[]
#lists to keep track of frame-by-frame angle and magnitude
allMags = []
allAngles = []

    
#loop through the video until the active frame reaches the end of our segment
while(cap.get(1) < endFrame):
    ret, frame2 = cap.read()
    #calculate new greyscale frame
    temp = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    #blur the frame as before
    next = cv.GaussianBlur(temp,(5,5),1)
    #cv.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 10, 3, 5, 1.1, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    
    """
    #this would convert the ang values from radian to degree
    angleOutput= ang*180/np.pi
    angleOutput = angleOutput.round()
    angleOutput = angleOutput.ravel()
    """
    #leave it in radians? just unravel the first matrix
    #otherwise comment out I guess
    angleOutput = ang.ravel()
    allAngles.append(angleOutput)
    
    #just loko at total magnitude
    magOutput = mag.round(4)
    magOutput = magOutput.ravel()
    allMags.append(magOutput)
    popPop.append(sum(magOutput))
    
    #this stuff shows the videos
    #note the hue calculation divides by 2 to get Left vs Right motion
    hsv[...,0] = ang*180/np.pi/2
    #normalize the values so they show up in the right ranges
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(20) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    
    prvs = next
    

cv.destroyAllWindows()
cv.waitKey(1)

#make a few matrices we'll append our individual frame comparisons to
#graph all motion across video
superAngle=np.concatenate(allAngles)
superMag=np.concatenate(allMags)
#calculating this a couple billion times would be time prohibitive
MagMean=superMag.mean()
MagStd=superMag.std()
MagOut=MagMean+MagStd

#we might also want to trrim small values to better see large values
#numpy.delete uses a list of indexes to remove them from arrays
#let's make a list of values below a cutoff and see what happens
trimList=[]

for val in range(len(superMag)):
    if superMag[val] < MagOut:
        trimList.append(val)
        
trimMag=np.delete(superMag,trimList)
trimAngle=np.delete(superAngle,trimList)
    

#scaling by max is no good due to weird random outliers
#better to scale by some multiple of standard deviation, I think
#keep consistent factor across frame graphs below to allow for easier comparison
scalingFactor= allMags[0].mean() + (3*allMags[0].std())


#need to produce a bunch of graphs for different frame comparisons
#put together individual frames in this loop
for frameComp in range(len(allMags)):
    #specify histogram bins for angles
    aBins = np.linspace(0,2*np.pi, 60)
    #bins for magnitude
    mBins = np.linspace(0,scalingFactor, 30)
    #now put everything together for a graph
    #need to label by frame comparison number
    if frameComp % 3 == 0:
        hist, _, _ = np.histogram2d(allAngles[frameComp], allMags[frameComp], bins=(aBins, mBins))
        A, M = np.meshgrid(aBins, mBins)
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        pc = ax.pcolormesh(A, M, hist.T, cmap="magma_r",label=str(frameComp))
        plt.title("Frame "+ str(startFrame+frameComp))
        fig.colorbar(pc)
        plt.show()
    
    

scalingFactor= superMag.mean() + (3*superMag.std())
#after doing frame-by-frame graphs, do a histogram of total overall motion
#angle bins first
aBins = np.linspace(0,2*np.pi, 60)
#magnitude bins next
#scale as above
mBins = np.linspace(0,scalingFactor, 30)
#finish up the graph
hist, _, _ = np.histogram2d(superAngle, superMag, bins=(aBins, mBins))
A, M = np.meshgrid(aBins, mBins)
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
pc = ax.pcolormesh(A, M, hist.T, cmap="magma_r")
plt.title("VidTotal")
fig.colorbar(pc)
plt.show()


#do plot of trimmed total info
scalingFactor= trimMag.mean() + (1*trimMag.std())
#after doing frame-by-frame graphs, do a histogram of total overall motion
#angle bins first
aBins = np.linspace(0,2*np.pi, 60)
#magnitude bins next
#scale as above
mBins = np.linspace(0,scalingFactor, 30)
#finish up the graph
hist, _, _ = np.histogram2d(trimAngle, trimMag, bins=(aBins, mBins))
A, M = np.meshgrid(aBins, mBins)
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
pc = ax.pcolormesh(A, M, hist.T, cmap="magma_r")
plt.title("trimTotal")
fig.colorbar(pc)
plt.show()

#last, some sort of line graph about total overall motion
#need to make a list of #s for individual frames
x=list(range(startFrame,endFrame-1))
#y-values are the magnitudes in popPop
y=popPop
# Plot line, line masks, then dots.
fig, ax = plt.subplots()
ax.plot(x, y, linestyle='-', color='black', linewidth=1, zorder=1)
ax.scatter(x, y, color='white', s=100, zorder=2)
ax.scatter(x, y, color='black', s=20, zorder=3)

# Remove axis lines.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set spine extent.
ax.spines['bottom'].set_bounds(min(x), max(x))
ax.spines['left'].set_bounds(225, 325)

# Reduce tick spacing.
x_ticks = list(range(min(x), max(x)+1, 2))
ax.xaxis.set_ticks(x_ticks)
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=25))
ax.tick_params(direction='in')

# Adjust lower limits to let data breathe.
ax.set_xlim([1950, ax.get_xlim()[1]])
ax.set_ylim([210, ax.get_ylim()[1]])

# Axis labels as a title annotation.
ax.text(1958, 320, 'Connecticut Traffic Deaths,\n1951 - 1959')



