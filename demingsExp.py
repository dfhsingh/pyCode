#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:32:39 2019

@author: kirandeshmukh
"""

import numpy as np
import matplotlib.pyplot as plt
import math, random
from datetime import datetime


# Code for the Smallest Enclosing Circle is taken from
# Project Nayuki - Library (Python) under the terms of 
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.# 
# Copyright (c) 2018 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
# 
# Data conventions: A point is a pair of floats (x, y).
# A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points.
# Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, 
# a circle of radius 0 is returned.
# 
# Initially: No boundary points known
def make_circle(points):
	# Convert to float and randomize order
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)
	
	# Progressively add points to circle or recompute circle
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c


# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left  = None
	right = None
	px, py = p
	qx, qy = q
	
	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or \
                        _cross_product(px, py, qx, qy, c[0], c[1]) > \
                        _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or \
                        _cross_product(px, py, qx, qy, c[0], c[1]) < \
                        _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c
	
	# Select which circle to return
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
	cx = (a[0] + b[0]) / 2.0
	cy = (a[1] + b[1]) / 2.0
	r0 = math.hypot(cx - a[0], cy - a[1])
	r1 = math.hypot(cx - b[0], cy - b[1])
	return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
	# Mathematical algorithm from Wikipedia: Circumscribed circle
	ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
	oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
	ax = a[0] - ox;  ay = a[1] - oy
	bx = b[0] - ox;  by = b[1] - oy
	cx = c[0] - ox;  cy = c[1] - oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) 
        + (cx*cx + cy*cy) * (ay - by)) / d
	y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) 
        + (cx*cx + cy*cy) * (bx - ax)) / d
	ra = math.hypot(x - a[0], y - a[1])
	rb = math.hypot(x - b[0], y - b[1])
	rc = math.hypot(x - c[0], y - c[1])
	return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= \
        c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by 
# (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    
# Code for the Experiment Starts Here

# Copyright (c) 2019 Kiran Deshmukh



def marbleDrop(funnel):
    sigmaXY = 2*sigma/np.sqrt(2.0)
    marble = [0, 0]
    marble[0] = funnel[0] + sigmaXY*np.random.randn()
    marble[1] = funnel[1] + sigmaXY*np.random.randn()
    return marble


def funnelShift(funnelLast, marbleLast, ruleNo):
    if ruleNo == 1:
        funnelNow = [funnelLast[0], funnelLast[1]]
    elif ruleNo == 2:
        funnelNow = [funnelLast[0] - marbleLast[0],
                     funnelLast[1] - marbleLast[1]]
    elif ruleNo == 3:
        funnelNow = [-marbleLast[0], -marbleLast[1]]
    elif ruleNo == 4:
        funnelNow = [marbleLast[0], marbleLast[1]]
    else:
        print("Error in Rule Number.")
        funnelNow = [0, 0]
    return funnelNow


def runExperiment():

    # Initialize
    i = 0
    funnelPos = [0, 0]
    marblePos = [0, 0]

    # Loop
    while i < iDrops:
        i += 1
        marblePos = marbleDrop(funnelPos)
        Points[i-1, :] = marblePos
        plt.plot(marblePos[0], marblePos[1], 'ro')
        funnelPos = funnelShift(funnelPos, marblePos, iRule)

    # Calculations
    Radius[:, 0] = np.sqrt(Points[:, 0]**2 + Points[:, 1]**2)
    meanPoint = [sum(Points[:, 0])/iDrops, sum(Points[:, 1])/iDrops]
    Rmax = max(Radius[:, 0])
    Rave =sum(Radius[:, 0])/iDrops
    sigmaX = np.std(Radius[:, 0])
    return Rmax, Rave, sigmaX, funnelPos, meanPoint


def printOutResults():
    
    # Messaging
    briefRule = [" ", \
    "Funnel will be adjusted based on the previous result.", \
    "Funnel will be adjusted relative to the Target," \
    " based on the previous result.", \
    "Funnel will be placed where the last marble fell."]
    print(briefRule[iRule-1])
    start_time = datetime.now()
    if iRule == 1:
        print("Wait! Dropping marbles.")
    else:
        print()
        print("Wait! Dropping marbles, and adjusting the funnel.")

    # Create the Figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Run the Experiment
    Rmax, Rave, sigmaX, funnelPosFinal, centPoint = runExperiment()
    
    # Mean Point from the Target
    Rcent = np.sqrt(centPoint[0]**2 + centPoint[1]**2)
    
    # Make the Enclosing Circle.
    points = Points.tolist()
    xCentCircle, yCentCircle, circleRad = make_circle(points)
    circleDia = 2.0*circleRad

    # Find the Elapsed Time.
    dt = datetime.now() - start_time
    timeElapsedSec = (dt.days*24*60*60 + dt.seconds) + dt.microseconds/1000000
    
    # Set the Scale
    yAxis = 8*sigma
    if Rmax > yAxis:
        yAxis = Rmax
    xAxis = 1.5*yAxis

    # Prinout the Chart.
    plt.title('The Spread of Marbles [Rule No. '+str(iRule)+']')
    plt.axis([-xAxis, xAxis, -yAxis, yAxis])
    plt.grid(True)
    plt.plot(0, 0, 'bs')
    plt.plot(centPoint[0], centPoint[1], 'gs')
    plt.plot([-xAxis, xAxis], [0, 0], color='blue')
    plt.plot([0, 0], [-yAxis, yAxis])
    enclosingCircle = plt.Circle((xCentCircle, yCentCircle), circleRad,
                                 color='g', fill=False)
    ax.add_patch(enclosingCircle)
    plt.show(block=False)
    
    # Printout the Results.
    print()
    if iRule != 1:
        print("NOTE THE SCALE OF THE GRAPH WHILE COMPARING THE SPREAD",
               "WITH OTHER GRAPHS!")
        print()
    print("Rule No.: ", iRule)
    print()
    print("Farthest Marble from the Target  : %8.4f" % Rmax)
    print("Average Distance from the Target : %8.4f" % Rave)
    print("Mean Point from the Target       : %8.4f" % Rcent)
    print("Diameter of the Enclosing Circle : %8.4f" % circleDia)
    print("Standard Deviation               : %8.4f" % sigmaX)
    print("Final Position of the Funnel     :  [%9.4f, %9.4f]" 
          % (funnelPosFinal[0], funnelPosFinal[1]))
    print("No of Drops                      : ", iDrops)
    print("Time Taken to Run the Experiment : %6.3f seconds" % timeElapsedSec)
    if rule_1 and iRule != 1:
        print()
        if Rmax > RmaxLast:
            print("THE PERFORMANCE HAS WORSENED!")
        else:
            print("CONTINUED BAD PERFORMANCE!")
        print()
    return Rmax


def printTitle():
    print()
    print("                    DR. DEMING'S FUNNEL EXPERIMENT")
    print()
    print("A marble is dropped through a funnel on a sheet of paper, which")
    print("contains the target. Objective of the experiment is to hit the")
    print("target. The experimenter can move the funnel based on",
          "certain rules.")
    print()
    print("RULE 1: Funnel is aligned over the target.")
    print("        No attempt is made to move the funnel to improve the",
          "performance.")
    print("        This is the baseline experiment to compare with the other",
          "rules.")
    print()
    print("RULE 2: We examine the previous result and take countermeasure.")
    print("        If the marble was 1 mm off northeast from the target,",
          "we move")
    print("        the funnel to position it 1 mm southwest of where it last",
          "was.")
    print()
    print("RULE 3: A possible flaw in the Rule 2 was that we adjusted the",
          "funnel")
    print("        from its last position, rather than relative to the",
          "target.")
    print("        If the marble was 1 mm off northwest from the target, we",
          "now")
    print("        position the funnel 1 mm southwest from the target.")
    print()
    print("RULE 4: In an attempt to reduce the variability of the marble",
          "drops,")
    print("        we decide to allow the marble to fall where it wants to.",
          "We")
    print("        position the funnel over the last location of the marble,",
          "as that")
    print("        appears to be the tendency of where the marble tends",
          "to stop.")
    print()
    return


def printComments():
    
    # Comments depending on the Rule No.
    print ()
    if iRule == 1:
        if (rule_2 and rule_3 and rule_4):
            print("Not moving the funnel seems to be the best way to minimize",
                  "the spread")
            print("of the marbles, and the only strategy to be closest to the",
                  "target.")
        elif (rule_2 or rule_3 or rule_4):
            print("Not moving the funnel seems to be a better way to minimize",
                  "the spread")
            print("of the marbles. Other rules may be tried to determine the",
                  "best strategy.")
        else:
            print("The marbles do not appear to behave consistently.",
                  "Certainly, there must")
            print("be a better (smarter) way to position the funnel to",
                  "improve the pattern.")
    if iRule == 2:
        print("A common example is worker adjustments to machinery. A worker",
              "may be")
        print("working to make a unit of uniform weight. If the last item was",
              "2 kg")
        print("underweight,increase the setting for the amount of material in",
              "the next")
        print("item by 2 kg.")
    if iRule == 3:
        print("We see Rule 3 at work in systems where two parties react to",
              "each other’s")
        print("actions. Their goal is to maintain parity. If one country",
              "increases its")
        print("nuclear arsenal, the rival country increases its arsenal to",
              "maintain the")
        print("perceived balance.")
    if iRule == 4:
        print("A common example of Rule 4 is when we want to cut lumber to",
              "a uniform")
        print("length. We use the piece we just cut in order to measure the",
              "location")
        print("of the next cut.")
        print()
        print("Other examples include adjusting starting time of the next",
              "meeting based")
        print("upon actual starting time of the last meeting, and a",
              "worker training")
        print("other worker who then trains another, and so forth.")
    return


def printEndNote():
    print()
    print("CONCLUSIONS OF THE DEMING'S FUNNEL EXPERIMENT:")
    print()
    print("Rules 2, 3, and 4 are all examples of process 'tampering'.")
    print()
    print("Rule 2 leads to a uniform circular pattern, whose size is about",
          "40% bigger")
    print("than the Rule 1 circle.")
    print()
    print("Rules 3 and 4 tend to “blow up”.")
    print()
    print("In Rule 3, results swing back and forth with greater and greater",
          "oscillations")
    print("from the target. In Rule 4, the funnel follows a drunken walk off",
          "the edge")
    print("of the table.")
    print()
    print("Rules 3 and 4 represent unstable systems, with over-corrections",
          "tending to occur.")
    print()
    return


# MAIN PROGRAM

# Set the Constants
iDrops = 1000
sigma = 1.0

# Initialize Program Parameters
Points = np.ndarray(shape=(iDrops, 2), dtype=float)
Radius = np.ndarray(shape=(iDrops, 1), dtype=float)
contRun = True
RmaxLast = 0.0
rule_1, rule_2, rule_3, rule_4 = False, False, False, False


printTitle()

# Input the Rule Number and Conduct the Experiment.
while contRun:
    
    # Get a valid Input.
    inputWrong = True
    while inputWrong:
        ruleInput = input("Enter the Rule Number [1/2/3/4], or q to quit: ")
        if ruleInput in ["1", "2", "3", "4", "q", "Q"]:
            inputWrong = False
    
    # Quitting?
    if ruleInput == "q" or ruleInput == "Q":
        contRun = False
        if (rule_1 and rule_2 and rule_3 and rule_4):
            printEndNote()
            print("Close the Charts to end.")
            plt.show()
        else:
            Ans = input("Are you sure to quit? ")
            if (Ans == "y" or Ans == "Y"):
                if (rule_1 or rule_2 or rule_3 or rule_4):
                    print("Close the Chart(s) to end.")
                    plt.show()
                print("Thank you.")
            else:
                contRun = True
    
    # Want to Run the Experiment.
    else:
        iRule = int(ruleInput)
        
        # Keeping Track of the Rules Used.
        if iRule == 1:
            rule_1 = True
        if iRule == 2:
            rule_2 = True
        if iRule == 3:
            rule_3 = True
        if iRule == 4:
            rule_4 = True
        
        # Run the Experiment.
        RmaxLast = printOutResults()
        printComments()
