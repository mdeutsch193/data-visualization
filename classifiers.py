# Martin Deutsch
# Template by Bruce Maxwell
# Spring 2017
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np
import scipy.cluster.vq as vq

class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type

    def type(self, newtype = None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def confusion_matrix( self, truecats, classcats ):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.

        '''
        
        # initialize matrix
        numcats = np.max(classcats)+1
        matrix = np.matrix( np.zeros((numcats, numcats)) )
        # loop through each data point
        for n in range(truecats.shape[0]):
            t = truecats[n, 0]
            c = classcats[n, 0]
            matrix[c, t] += 1
        
        return matrix

    def confusion_matrix_str( self, cmtx ):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        
        s = '\nConfusion Matrix:\n'
        s += 'Actual->'
        for i in range(cmtx.shape[0]):
            s += ('%8s %d') % ('Class', i)
        s += '\n'
        for i in range(cmtx.shape[0]):
            s += 'Class %d:' % (i)
            for j in range(cmtx.shape[1]):
                s += "%10d" % (cmtx[i, j])
            s += '\n'
            
        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)



class NaiveBayes(Classifier):
    '''NaiveBayes implements a simple NaiveBayes classifier using a
    Gaussian distribution as the pdf.

    '''

    def __init__(self, dataObj=None, headers=[], categories=None):
        '''Takes in a Data object with N points, a set of F headers, and a
        matrix of categories, one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'Naive Bayes Classifier')
        
        self.headers = headers
        self.num_classes = 0
        self.num_features = 0
        self.class_labels = np.matrix([])
        self.class_means = np.matrix([])
        self.class_vars = np.matrix([])
        self.class_scales = np.matrix([])
        
        if dataObj != None:
            self.build(dataObj.get_data(headers), categories)

    def build( self, A, categories ):
        '''Builds the classifier given the data points in A and the categories'''
        
        # figure out how many categories there are and get the mapping
        unique, mapping = np.unique( np.array(categories.T), return_inverse=True )
        
        # create the matrices for the means, vars, and scales
        means = np.matrix( np.zeros((unique.shape[0], A.shape[1])) )
        vars = np.matrix( np.zeros((unique.shape[0], A.shape[1])) )
        scales = np.matrix( np.zeros((unique.shape[0], A.shape[1])) )
		
        # compute the means/vars/scales for each class
        for i in range(unique.shape[0]):
            means[i, :] = np.mean(A[(mapping==unique[i]),:], axis=0)
            vars[i, :] = np.var(A[(mapping==unique[i]),:], axis=0, ddof=1)
            scales[i, :] = 1/np.sqrt(vars[i, :]*2*np.pi)
        	
        self.class_means = means
        self.class_vars = vars
        self.class_scales = scales
    		
        # store other necessary information
        self.num_classes = unique.shape[0]
        self.num_features = A.shape[1]
        self.class_labels = categories

        return

    def classify( self, A, return_likelihoods=False ):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_likelihoods
        is True, it also returns the NxC likelihood matrix.'''
        
        # error check to see if A has the same number of columns as
        # the class means
        if A.shape[1] != self.class_means.shape[1]:
            print "Must be same matrix used in building classifier"
            return
        	
        # make a matrix that is N x C to store the probability of each
        # class for each data point
        P = np.matrix( np.zeros((A.shape[0], self.num_classes)) )
		
        # calculate the probabilities by looping over the classes
        for i in range(self.num_classes):
            # probability is given by the product of Gaussian distributions 
            P[:, i] = np.prod(np.multiply(self.class_scales[i], 
                                np.exp(-np.square(A-self.class_means[i])/
                                (2*self.class_vars[i]))), axis=1)
        
        # calculate the most likely class for each data point
        cats = np.argmax(P, axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.class_means[i,:]) + "\n"
            s += 'Var   : ' + str(self.class_vars[i,:]) + "\n"
            s += 'Scales: ' + str(self.class_scales[i,:]) + "\n"

        s += "\n"
        return s
        
    def write(self, filename):
        '''Writes the Bayes classifier to a file.'''
        text_file = open(filename, "w")
        text_file.write(self.__str__())
        text_file.close()
        return

    def read(self, filename):
        '''Reads in the Bayes classifier from the file'''
        self.num_classes = 0
        means = []
        vars = []
        scales = []
        
        text_file = open(filename, "r")
        lines = text_file.readlines()
        for line in lines:
        	if "Class " in line:
        		self.num_classes += 1
        	if "Mean" in line:
        		means.append(np.matrix(line[line.index("[[")+2 : line.index("]]")]))
        	if "Var" in line:
        		vars.append(np.matrix(line[line.index("[[")+2 : line.index("]]")]))
        	if "Scales" in line:
        		scales.append(np.matrix(line[line.index("[[")+2 : line.index("]]")]))
        text_file.close()
        
        self.means = np.vstack(means)
        self.vars = np.vstack(vars)
        self.scales = np.vstack(scales)
        self.num_features = self.means.shape[0]
        
        return

    
class KNN(Classifier):

    def __init__(self, dataObj=None, headers=[], categories=None, K=None):
        '''Take in a Data object with N points, a set of F headers, and a
        matrix of categories, with one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')
        
        self.headers = headers
        self.num_classes = 0
        self.num_features = 0
        self.class_labels = np.matrix([])
        self.exemplars = []
        
        if dataObj != None:
            self.build(dataObj.get_data(headers), categories, K)

    def build( self, A, categories, K = None ):
        '''Builds the classifier given the data points in A and the categories'''

        # figure out how many categories there are and get the mapping
        unique, mapping = np.unique( np.array(categories.T), return_inverse=True )
        
        # for each category i, build the set of exemplars
        for i in range(unique.shape[0]):
            if K == None:
                self.exemplars.append(A[(mapping==unique[i]),:])
            else:
                codebook, bookerror = vq.kmeans(A[(mapping==unique[i]),:], K)
                self.exemplars.append(codebook)
                    
        # store other necessary information
        self.num_classes = unique.shape[0]
        self.num_features = A.shape[1]
        self.class_labels = categories
        
        return

    def classify(self, A, K=3, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.

        The parameter K specifies how many neighbors to use in the
        distance computation. The default is three.'''

        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.matrix( np.zeros((A.shape[0], self.num_classes)) )
        
        # compute the distance between each data point and each class
        for i in range(self.num_classes):
            temp = np.matrix( np.zeros((A.shape[0], self.exemplars[i].shape[0])) )
            for e in range(self.exemplars[i].shape[0]):
                temp[:, e] = np.sqrt( np.sum(np.square(A - self.exemplars[i][e, :]), axis=1) )
            temp.sort(axis=1)
            D[:, i] = np.sum(temp[:, 0:K], axis=1)
        
        # calculate the most likely class for each data point
        cats = np.argmin(D, axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s


    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        text_file = open(filename, "w")
        s = "KNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Exemplars: \n'
            s += str(self.exemplars[i])
            s += '\n'
        text_file.write(s)
        text_file.close()
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        self.num_classes = 0
        self.exemplars = []
        exs = []
        
        text_file = open(filename, "r")
        lines = text_file.readlines()
        for line in lines:
        	if "Class " in line:
        		self.num_classes += 1
        	elif "[[" in line:
        		exs.append([ np.matrix(
                                line[line.index("[")+2 : line.index("]")]) ])
        	elif "[" in line:
        		exs[self.num_classes-1].append( np.matrix(
                                line[line.index("[")+1 : line.index("]")]) )
        text_file.close()

        for i in range(self.num_classes):
            exemplars = np.vstack(exs[i])
            self.exemplars.append(exemplars)
        self.num_features = self.exemplars[0].shape[0]
        
        return
        
class NN(KNN):

    def classify(self, A, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.'''

        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.matrix( np.zeros((A.shape[0], self.num_classes)) )
        
        # compute the distance between each data point and each class
        for i in range(self.num_classes):
            temp = np.matrix( np.zeros((A.shape[0], self.exemplars[i].shape[0])) )
            for e in range(self.exemplars[i].shape[0]):
                temp[:, e] = np.sqrt( np.sum(np.square(A - self.exemplars[i][e, :]), axis=1) )
            temp.sort(axis=1)
            D[:, i] = temp[:, 0]
        
        # calculate the most likely class for each data point
        cats = np.argmin(D, axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels