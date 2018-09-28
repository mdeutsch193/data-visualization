# Martin Deutsch
# Spring 2017
# CS 251 Project 9
#
# Classify Bird Arrival Data Set
#

import sys
import csv
import data
import classifiers

def main(argv):
    ''' Reads in a training set and a test set and builds a KNN classifer.
    Prints out confusion matrices and writes classifications 
    for test data to a CSV file. '''

    # usage
    if len(argv) < 3:
        print 'Usage: python %s <training data file> <test data file> <optional training category file> <optional test category file>' % (argv[0])
        exit(-1)

    # read the training and test sets
    dtrain = data.Data(argv[1])
    dtest = data.Data(argv[2])

   # get the categories and the training data A and the test data B
    if len(argv) > 4:
        traincatdata = data.Data(argv[3])
        testcatdata = data.Data(argv[4])
        traincats = traincatdata.get_data( [traincatdata.get_headers()[0]] )
        testcats = testcatdata.get_data( [testcatdata.get_headers()[0]] )
        A = dtrain.get_data( dtrain.get_headers() )
        B = dtest.get_data( dtest.get_headers() )
    else:
        # assume the categories are the last column
        traincats = dtrain.get_data( [dtrain.get_headers()[-1]] )
        testcats = dtest.get_data( [dtest.get_headers()[-1]] )
        A = dtrain.get_data( dtrain.get_headers()[:-1] )
        B = dtest.get_data( dtest.get_headers()[:-1] )

    # create KNN classifier
    knnc = classifiers.KNN()

    # build the classifier using the training data
    knnc.build( A, traincats )

    # use the classifier on the training data
    knnctraincats, knnctrainlabels = knnc.classify( A )
    print "For KNN (training data):"
    print knnc.confusion_matrix_str(knnc.confusion_matrix(traincats, knnctraincats))
	
    # use the classifier on the test data
    knnctestcats, knnctestlabels = knnc.classify( B )
    print "For KNN (test data):"
    print knnc.confusion_matrix_str(knnc.confusion_matrix(testcats, knnctestcats))
    
    # write test data to csv
    knncfile = open("knncOut.csv", 'w')
    writeFile = csv.writer(knncfile)
    if len(argv) > 4:
    	knncHeaders = dtest.get_headers()
    else:
    	knncHeaders = dtest.get_headers()[:-1]
    knncHeaders.append("Category")
    writeFile.writerow(knncHeaders)
    writeFile.writerow(["numeric"]*len(knncHeaders))
    for i in range(B.shape[0]):
        rowList = B[i, :].tolist()
        rowList[0].append(knnctestcats[i, 0])
        writeFile.writerow(rowList[0])
    knncfile.close()

    return

if __name__ == "__main__":
    main(sys.argv)