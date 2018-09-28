# Martin Deutsch 
# Project 8
# CS 251
# Spring 2017

import numpy as np
import csv

# create a class to read and store data from a file
class Data:
    
    # create fields to hold data and handle file
    def __init__(self, filename = None):
        # create and initialize fields for the class
        self.raw_headers = []
        self.raw_types = []
        self.raw_data = []
        self.header2raw = {}
        self.matrix_data = np.matrix([])
        self.headers = []
        self.header2matrix = {}
        
        # if given csv file, read file
        if filename != None and filename != '':
            self.read(filename)
    
    # read data from file into fields
    def read(self, filename):
        file = open(filename, 'rU')
        readFile = csv.reader(file)
        
        # store raw data
        self.raw_headers = readFile.next()
        for i in range(len(self.raw_headers)):
            self.raw_headers[i] = self.raw_headers[i].strip(' ')
            self.header2raw[self.raw_headers[i]] = i
        self.raw_types = readFile.next()
        for i in range(len(self.raw_types)):
            self.raw_types[i] = self.raw_types[i].strip(' ')
        for row in readFile:
            if '#' in row[0]:
                readFile.next()
            else:
                self.raw_data.append(row)
        file.close()

        # find columns of numeric data
        cols = []
        for c in range (len(self.raw_types)):
            if self.raw_types[c] == "numeric":
                self.headers.append(self.raw_headers[c])
                self.header2matrix[self.raw_headers[c]] = len(cols)
                col = []
                for row in self.raw_data:
                	try:
                		col.append(float(row[c]))
                	except ValueError:
                		col.append(-9999)
                cols.append(col)
        # add numeric columns to numpy matrix and transpose matrix
        self.matrix_data = np.matrix(cols)
        self.matrix_data = self.matrix_data.T
    
    # return list of headers
    def get_raw_headers(self):
        return self.raw_headers
    
    # return list of types
    def get_raw_types(self):
        return self.raw_types
    
    # return number of columns
    def get_raw_num_columns(self):
        return len(self.raw_headers)
    
    # return number of rows
    def get_raw_num_rows(self):
        return len(self.raw_data)
    
    # return the row of data for the given index
    def get_raw_row(self, index):
        return self.raw_data[index]
    
    # return the value in the given row under the given column header
    def get_raw_value(self, rowIndex, colHeader):
        return self.raw_data[rowIndex][self.header2raw[colHeader]]
    
    # print out raw data to command line
    def print_data(self):
        # find max column width
        col_width = 2
        for header in self.raw_headers:
            if len(header) > col_width:
                col_width = len(header)
        for row in self.raw_data:
            for val in row:
                if len(val) > col_width:
                    col_width = len(val)
        col_width += 2
        # print table
        for header in self.raw_headers:
            print header.ljust(col_width),
        print
        for row in self.raw_data:
            for val in row:
                print val.ljust(col_width),
            print

    # return list of headers of columns with numeric data
    def get_headers(self):
        return self.headers
    
    # return the number of columns of numeric data
    def get_num_columns(self):
        return len(self.headers)
        
    # return the row of numeric data for the given index
    def get_row(self, index):
        return self.matrix_data[index, :].tolist()[0]
        
    # return the numeric value for the given row index and column header
    def get_value(self, rowIndex, colHeader):
        return self.matrix_data[rowIndex, self.header2matrix[colHeader]]
    
    # return the data for the given columns and rows
    def get_data(self, colHeaders, rows=None):
        if rows == None:
            rows = range(len(self.matrix_data))
        # initialize matrix with first column
        colIndex = self.header2matrix[colHeaders[0]]
        matrix = self.matrix_data[rows, colIndex:colIndex+1]
        # add the other columns
        for i in range(1, len(colHeaders)):
            colIndex = self.header2matrix[colHeaders[i]]
            matrix = np.hstack( (matrix, self.matrix_data[rows, colIndex:colIndex+1]) )
        return matrix
    
    # add a numeric column of data to the matrix
    def add_column(self, header, col):
        self.raw_headers.append(header)
        self.header2raw[header] = len(self.raw_headers)-1
        self.raw_types.append("numeric")
        self.headers.append(header)
        self.matrix_data = np.hstack((self.matrix_data, col[:, 0]))
        self.header2matrix[header] = len(self.headers)-1
        for i in range(len(self.raw_data)):
            self.raw_data[i].append(str(col[i, 0]))
            
    # add a numeric row of data to the matrix
    def add_row(self, row):
        self.matrix_data = np.vstack((self.matrix_data, row))
        rawrow = []
        for i in range(row.shape[1]):
            rawrow.append(str(row[0, i]))
        self.raw_data.append(rawrow)
    
    # delete a row from the data object
    def delete_row(self, row):
    	self.matrix_data = np.delete(self.matrix_data, (row), axis=0)
    	del self.raw_data[row]
    
    # write numeric data to csv file
    def write(self, filename, headers = ''):
        if headers == '':
            headers = self.headers
        types = ["numeric"]*len(headers)
        file = open(filename+".csv", 'w')
        writeFile = csv.writer(file)
        writeFile.writerow(headers)
        writeFile.writerow(types)
        data = self.get_data(headers)
        for row in data:
            rowList = row.tolist()
            writeFile.writerow(rowList[0])
        file.close()
        

# create a class to hold the results of Principal Component Analysis
class PCAData(Data):
    
    # initalize parent fields and new fields for PCA values
    def __init__(self, headers, projData, evals, evecs, means): 
        # store headers
        self.raw_headers = []
        for i in range(len(headers)):
            self.raw_headers.append("e" + str(i))
        self.header2raw = {}
        for i in range(len(self.raw_headers)):
            self.header2raw[self.raw_headers[i]] = i
        self.headers = self.raw_headers
        self.header2matrix = {}
        for i in range(len(self.headers)):
            self.header2matrix[self.headers[i]] = i
        
        # store data
        self.raw_types = ['numeric']*len(headers)
        self.raw_data = []
        for r in range(projData.shape[0]):
            row = []
            for c in range(projData.shape[1]):
                row.append(str(projData[r, c]))
            self.raw_data.append(row)
        
        self.matrix_data = projData;
        self.evals = np.matrix(evals)
        self.evecs = evecs
        self.means = means
        self.pHeaders = headers
        
    # returns the eigenvalues as a single-row numpy matrix
    def get_eigenvalues(self):
        return self.evals

    # returns the eigenvectors as a numpy matrix with the eigenvectors as rows
    def get_eigenvectors(self):
        return self.evecs
        
    # returns the means for each column in the data as a single row numpy matrix
    def get_data_means(self):
        return self.means

    # returns the list of the headers used to generate the projected data
    def get_data_headers(self):
        return self.pHeaders
    
# create a class to hold results of cluster analysis
class ClusterData(Data):
    
    # initalize parent fields
    def __init__(self, headers, data):
        # store headers
        self.raw_headers = []
        for i in range(len(headers)):
            self.raw_headers.append(headers[i])
        self.header2raw = {}
        for i in range(len(self.raw_headers)):
            self.header2raw[self.raw_headers[i]] = i
        self.headers = []
        for i in range(len(headers)):
            self.headers.append(headers[i])
        self.header2matrix = self.header2raw

        # store data
        self.raw_types = ['numeric']*len(headers)
        self.raw_data = []
        for r in range(data.shape[0]):
            row = []
            for c in range(data.shape[1]):
                row.append(str(data[r, c]))
            self.raw_data.append(row)
        self.matrix_data = data;