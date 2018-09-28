# Martin Deutsch 
# Project 8
# CS 251
# Spring 2017

import sys
import numpy as np
import scipy.stats
import scipy.cluster.vq as vq
import data
    
# return a list of two element lists of mins and maxes for each column
def data_range(data, colHeaders):
    numData = data.get_data(colHeaders)
    mins = np.min(numData, axis=0)
    maxs = np.max(numData, axis=0)
    ranges = [ [round(mins[0, 0], 2), round(maxs[0, 0], 2)] ]
    for i in range(1, len(colHeaders)):
        ranges.append([round(mins[0, i], 2), round(maxs[0, i], 2)])
    return ranges

# return a list of the means of the specifed columns
def mean(data, colHeaders):
    numData = data.get_data(colHeaders)
    meanMatrix = np.mean(numData, axis=0)
    means = meanMatrix.tolist()
    for i in range(len(means[0])):
        means[0][i] = round(means[0][i], 2)
    return means[0]

# return a list of the standard deviations of the specified columns
def stdev(data, colHeaders):
    numData = data.get_data(colHeaders)
    stdMatrix = np.std(numData, axis=0, ddof=1)
    stds = stdMatrix.tolist()
    return stds[0]

# return matrix with normalized columns so min is 0 and max is 1
def normalize_columns_separately(data, colHeaders):
    numData = data.get_data(colHeaders)
    mins = np.min(numData, axis=0)
    maxs = np.max(numData, axis=0)
    tmp = numData - mins
    range = maxs - mins
    return tmp / np.matrix(range, dtype=float)

# return normalized matrix so min is 0 and max is 1
def normalize_columns_together(data, colHeaders):
    numData = data.get_data(colHeaders)
    min = np.min(numData)
    max = np.max(numData)
    tmp = numData - min
    range = max - min
    return tmp / np.matrix(range, dtype=float)

# return a list of the medians of the specifed columns
def median(data, colHeaders):
    numData = data.get_data(colHeaders)
    medianMatrix = np.median(numData, axis=0)
    medians = medianMatrix.tolist()
    return medians

# execute linear regression
def linear_regression(data, ind, dep):
    # initialize y and A matrices
    y = np.matrix(data.get_data([dep]))
    A = np.matrix(data.get_data(ind))
    A = np.hstack( (A, np.matrix(np.ones(data.get_raw_num_rows())).T) )
    
    # covariance matrix of the independent data, used for computing 
    # the standard error later
    AAinv = np.linalg.inv( np.dot(A.T, A))
    
    # Solves the equation y = Ab for b
    x = np.linalg.lstsq( A, y )
    # Solution that provides the best fit regression
    b = x[0]
    # Number of data points
    N = y.shape[0]
    # number of coefficients
    C = b.shape[0]
    # number of degrees of freedom of the error
    df_e = N-C
    # number of degrees of freedom of the model fit
    df_r = C-1

    # the error of the model prediction (the vertical difference between
    # the regression line and the data)
    error = y - np.dot(A, b)

    # the sum squared error (sum of errors divided by the number of 
    # degrees of freedom of the error)
    sse = np.dot(error.T, error) / df_e # 1x1 matrix

    # the standard error (the square root of the diagonals of 
    # the sum-squared error multiplied by the inverse covariance matrix)
    stderr = np.sqrt( np.diagonal( sse[0, 0] * AAinv ) ) # Cx1 vector

    # the t-statistic for each independent variable 
    # (each coefficient of the fit divided by the standard error)
    t = b.T / stderr

    # probability of the coefficient indicating a random relationship (slope = 0)
    # (cumulative distribution function of the t distribution
    #  multiplied by 2 to get the 2-sided tail)
    p = 2*(1 - scipy.stats.t.cdf(abs(t), df_e))

    # the r^2 coefficient indicating the quality of the fit
    r2 =  1 - error.var() / y.var()
    
    return b, sse[0, 0], r2, t, p

# execute PCA using SVD
def pca(d, headers, normalize=True):
    
    # assign to A the desired data
    if (normalize):
        A = normalize_columns_separately(d, headers)
    else:
        A = d.get_data(headers)
  
    # assign to m the mean values of the columns of A
    m = np.mean(A, axis=0)

    # assign to D the difference matrix A - m
    D = A - m

    # assign to U, S, and V the results of SVD
    U, S, V = np.linalg.svd(D, full_matrices=False)

    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    #   divided by the degrees of freedom (N-1). The values are sorted.
    evals = S*S/(A.shape[0]-1)

    # project the data onto the eigenvectors. 
    #   The eigenvectors match the order of the eigenvalues.
    proj = (V*D.T).T

    # create and return a PCA data object with the headers, projected data, 
    # eigenvectors, eigenvalues, and mean vector.
    pca = data.PCAData(headers, proj, evals, V, m)
    return pca

# execute k-means clustering using numpy
# Takes in a Data object, a set of headers, and the number of clusters to create
# Computes and returns the codebook, codes, and representation error
def kmeans_numpy( d, headers, K, whiten = True ):
    A = d.get_data(headers)
    if whiten:
        W = vq.whiten(A)
    else:
        W = A
    codebook, bookerror = vq.kmeans(W, K)
    codes, error = vq.vq(W, codebook)
    return codebook, codes, error

# initialize cluster means
def kmeans_init( A, K, categories=None ):
    if categories == None:
        pts = [ np.random.randint(0, A.shape[0]) ]
        while len(pts) < K:
            rand = np.random.randint(0, A.shape[0])
            if rand not in pts:
                pts.append(rand)
        means = A[pts[0], :]
        for i in range(1, K):
            means = np.vstack( (means, A[pts[i], :]) )
   
    else:
        pts = []
        for c in range(categories.shape[0]):
            if categories[c] == 0:
                pts.append(c)
        cat = A[pts[0], :]
        for i in range(1, len(pts)):
            cat = np.vstack( (cat, A[pts[i], :]) )
        means = np.mean(cat, axis=0)
        
        for n in range(1, K):
            pts = []
            for c in range(categories.shape[0]):
                if categories[c] == n:
                    pts.append(c)
            cat = A[pts[0], :]
            for i in range(1, len(pts)):
                cat = np.vstack( (cat, A[pts[i], :]) )
            means = np.vstack( (means, np.mean(cat, axis=0)) )
            
    return means

# classify data points by closest mean
def kmeans_classify(A, means, dmetric=2):
    pt = A[0, :]
    if dmetric == 1:
        minDist = np.sum( abs(pt - means[0, :]) )
    elif dmetric == 2:
        minDist = np.sum( (pt - means[0, :])*(pt - means[0, :]).T )**.5
    else:
        minDist = np.max( abs(pt-means[0, :]) )
    id = 0
    for i in range(1, means.shape[0]):
        if dmetric == 1:
            dist = np.sum( abs(pt-means[i, :]) )
        elif dmetric == 2:
            dist = np.sum( (pt-means[i, :])*(pt-means[i, :]).T )**.5
        else:
            dist = np.max ( abs(pt-means[i, :]) )
        if dist < minDist:
            minDist = dist
            id = i
    distances = np.matrix(minDist)
    ids = np.matrix([id])
    
    for n in range(1, A.shape[0]):
        pt = A[n, :]
        if dmetric == 1:
             minDist = np.sum( abs(pt - means[0, :]) )
        elif dmetric == 2:
            minDist = np.sum( (pt - means[0, :])*(pt - means[0, :]).T )**.5
        else:
            minDist = np.max( abs(pt-means[0, :]) )
        id = 0
        for i in range(1, means.shape[0]):
            if dmetric == 1:
                dist = np.sum( abs(pt-means[i, :]) )
            elif dmetric == 2:
                dist = np.sum( (pt-means[i, :])*(pt-means[i, :]).T )**.5
            else:
                dist = np.max( abs(pt-means[i, :]) )
            if dist < minDist:
                minDist = dist
                id = i
        distances = np.vstack( (distances, np.matrix(minDist)) )
        ids = np.vstack( (ids, np.matrix([id])) )
    
    return ids, distances

# core k-means algorithm
def kmeans_algorithm(A, means, metric):
    # set up some useful constants
    MIN_CHANGE = 1e-7
    MAX_ITERATIONS = 100
    D = means.shape[1]
    K = means.shape[0]
    N = A.shape[0]

    # iterate no more than MAX_ITERATIONS
    for i in range(MAX_ITERATIONS):
        # calculate the codes
        codes, errors = kmeans_classify( A, means, metric )

        # calculate the new means
        newmeans = np.zeros_like( means )
        counts = np.zeros( (K, 1) )
        for j in range(N):
            newmeans[codes[j,0],:] += A[j,:]
            counts[codes[j,0],0] += 1.0

        # finish calculating the means, taking into account possible zero counts
        for j in range(K):
            if counts[j,0] > 0.0:
                newmeans[j,:] /= counts[j, 0]
            else:
                newmeans[j,:] = A[random.randint(0,A.shape[0]),:]

        # test if the change is small enough
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call classify with the final means
    codes, errors = kmeans_classify( A, means, metric )

    # return the means, codes, and errors
    return (means, codes, errors)

# Takes in a Data object, a set of headers, and the number of clusters to create
# Computes and returns the codebook, codes and representation errors. 
# If given an Nx1 matrix of categories, it uses the category labels 
#   to calculate the initial cluster means.
def kmeans(d, headers, K, metric=2, whiten=True, categories=None):
    A = d.get_data(headers)
    if whiten:
        W = vq.whiten(A)
    else:
        W = A
    codebook = kmeans_init(W, K, categories)
    codebook, codes, errors = kmeans_algorithm(W, codebook, metric)
    return codebook, codes, errors