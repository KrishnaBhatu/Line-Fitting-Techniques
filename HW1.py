'''
 * BSD 3-Clause License
 * @copyright (c) 2019, Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * @file    HW1.py
 * @author  Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * @version 1.0
 * @brief Demonstration of Principles of Linear Estimation
 *
 * @section DESCRIPTION
 *
 * Python source code implementation of line fitting of 2D data points using methods like:
        1) Linear Least Squares (Least Squares and Total Least Squares)
        2) Linear Square Estimation with Regularization
        3) Outliers rejection using RANSAC
 '''

import pickle 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import sqrt
from math import atan2
from math import pi
import random
#This function resizes all the plots for better display
plt.rcParams["figure.figsize"] = (10,10)
'''
Function Description of findEigenCovariance:
    @param data: It contains the 2D points from the pickle data file
    @param title: It is a string which gives the title to the plot according to the data set number

    @return x: It contains an array of x co-ordinates of the data points
    @return y: It contains an array of y co-ordinates of the data points
    @return values: It contains the eigen values of the data points
    @return array: It contains the 2D data points in numpy array format

    @brief This function calculates the co-variance matrix for the given data set of 2D points
           and calculates the eigen value and eigen vectors for the same.
'''
def  findEigenCovariance(data, title):
    array = np.array(data)
    x,y = array.T
    covarianceMatrix = np.cov(array, rowvar = False)
    values, vectors = LA.eig(covarianceMatrix)
    plotEigen(title, x,y,values,vectors)
    return x , y, values, array
'''
Function Description of plotEigen:
    @param title: It is a string which gives the title to the plot according to the data set number
    @param x: It contains an array of x co-ordinates of the data points
    @param y: It contains an array of y co-ordinates of the data points
    @param values: It contains the eigen values of the data points
    @param vectors: It contains the eigen vectors of the data points

    @return void

    @brief This function plots the scatter plot for the data set and also the eigen vectors
           for the data which reprents the direction of the variance of the data
'''
def plotEigen(title, x, y, values, vectors):
    plt.figure(title)
    plt.scatter(x, y)
    n = y.shape[0]
    mean = [np.sum(x)/n], [np.sum(y)/n] # mean of data
    QV1 = plt.quiver(*mean, vectors[0,0], vectors[1,0], color=['g'],scale = 1/50, angles = "xy", scale_units = "xy")
    QV2 = plt.quiver(*mean, vectors[0,1], vectors[1,1], color=['b'], scale = 1/25, angles = "xy", scale_units = "xy")
    plt.quiverkey( Q = QV1, X = 0.85, Y = 0.05, U = 1, label = 'Principle Eigen vector', coordinates='axes')
    plt.quiverkey( Q = QV2, X = 0.85, Y = 0.01, U = 1, label = 'Secondary Eigen vector', coordinates='axes')

'''
Function Description of leastSquares:
    @param x: It contains an array of x co-ordinates of the data points
    @param y: It contains an array of y co-ordinates of the data points
    @param lambda: It is the magnitude of penalty used for Linear Square Estimation with Regularization

    @return void

    @brief This function calculates and plots the lines which fits the data set by the methods of
           Least Squares(Minimize Vertical Error), Least Squares with Regularization and
           Total Least Squares(Minimize Orthogonal Error)
'''
def leastSquares(x , y,  lamda):
    #LS
    onlyOnes = np.ones(x.shape[0])
    A = np.column_stack((x,onlyOnes))
    xStar = np.matmul(np.matmul(np.transpose(A),y),np.linalg.inv(np.matmul(np.transpose(A),A)))
    xOfLine = np.linspace(-100,100,100)
    yOfLSLine = xStar[0]*xOfLine+xStar[1]
    
    #LS + Regularization
    xStar2 = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(A),A),lamda*np.identity(2))) , np.matmul(np.transpose(A),y))
    yOfRLSLine = xStar2[0]*xOfLine+xStar2[1]

    #TLS
    n = y.shape[0]
    yBar = np.sum(y)/n
    ysquareBar = np.sum(np.square(y))
    xBar = np.sum(x)/n
    xsquareBar = np.sum(np.square(x))
    denominator= n*xBar*yBar - np.sum(np.multiply(x,y))
    numerator = (ysquareBar - n*yBar*yBar) - (xsquareBar - n*xBar*xBar) 
    B = 0.5*(numerator/denominator)
    m = -B + sqrt(B*B + 1)
    c = yBar - m*xBar
    yOfTLSLine = m*xOfLine+c
    plotLines(xOfLine, yOfLSLine, yOfRLSLine,yOfTLSLine)
'''
Function Description of plotLines:
    @param xofLine: It contains x co-ordinate data of the line to plot
    @param yofLSine: It contains y co-ordinates data of the line by Least Square to plot
    param yofRLSine: It contains y co-ordinates data of the line by Least Square with Regularization to plot
    @param yofTLSine: It contains y co-ordinates data of the line by Total Least Square to plot

    @return void

    @brief This function plots the lines which fits the data set by the methods of
           Least Squares(Minimize Vertical Error), Least Squares with Regularization and
           Total Least Squares(Minimize Orthogonal Error). It is called by leastSquares function
'''
def plotLines(xOfLine, yOfLSLine, yOfRLSLine,yOfTLSLine):
    plt.plot(xOfLine, yOfLSLine, '-r', linewidth=2.0, label="LS")
    plt.plot(xOfLine, yOfRLSLine, '-y', linewidth=2.0, label="LS + Regularization")
    plt.plot(xOfLine, yOfTLSLine, '-k', linewidth=2.0, label="TLS")
'''
Function Description of ransac:
    @param array: It contains the 2D data points in numpy array format
    @param threshold: It is the threshold set for the RANSAC algorithm(Default = 5)
    @param iteration: It is the number of iterations for RANSAC algorithm to reach to desired output(Default = 100)
    
    @return void

    @brief This function calculates and plots the lines which fits the data set by the methods of
           RANSAC
'''
def ransac(array, threshold = 5, iterations = 100):
    prevInliners = 0
    for i in range(0,iterations):
        point1 = random.choice(array)
        point2 = random.choice(array)
        if(abs(point2[0] - point1[0]) > 0.01):
            slope  = (point2[1] - point1[1])/(point2[0] - point1[0]) 
            intercept =  point1[1] - slope*point1[0]
            inliners = 0
            for point in array:
                compare = point[1] - slope*point[0]
                if((compare >(intercept - threshold)) and (compare < (intercept+threshold))):
                    inliners += 1
            if (inliners > prevInliners):
                prevInliners = inliners
                m = slope
                c = intercept
        else:
            pass
    a = np.linspace(-100,100,100)
    b = m*a+c
    plt.plot(a, b, color = '#c61bbe', linewidth=2.0, label="RANSAC")

# Data 1 
data  = pickle.load( open( "data1_new.pkl", "rb" ) )
x , y, values, array1 = findEigenCovariance(data, "Data 1")
leastSquares(x ,y, 2)
ransac(array = array1)
plt.legend()
plt.show(block=False)

# Data 2
data = pickle.load( open( "data2_new.pkl", "rb" ) )
x , y, values, array1 = findEigenCovariance(data, "Data 2")
leastSquares(x ,y, values)
ransac(array = array1)
plt.legend()
plt.show(block=False)

# Data 3
data  = pickle.load( open( "data3_new.pkl", "rb" ) )
x , y, values, array1 = findEigenCovariance(data, "Data 3")
leastSquares(x ,y, values)
ransac(array = array1)
plt.legend()

plt.show()
