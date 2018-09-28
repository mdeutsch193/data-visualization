# Martin Deutsch
# Spring 2017

import data
import numpy

if __name__ == '__main__':
	d = data.Data( "Arrival Dates Master - 1994-2014.csv" )
	
	# categorize by species
	speciesCats = []
	speciesDict = {}
	c = 0
	for i in range(d.get_raw_num_rows()):
		species = d.get_raw_value(i, 'Species' )
		if speciesDict.has_key(species):
			speciesCats.append(speciesDict[species])
		else:
			speciesCats.append(c)
			speciesDict[species] = c
			c += 1
	d.add_column("speciesCat", numpy.matrix(speciesCats).T)
	
	# add category list for biophysical region
	regionCats = []
	regionDict = {
        "One":1, "Two":2, "Three":3, "Four":4, "Five":5, "Six":6, "Seven":7, 
        "Eight":8, "Nine":9, "Ten":10, "Eleven":11, "Twelve":12, "Thirteen":13, 
        "Fourteen":14, "Fifteen":15
    }
	for i in range(d.get_raw_num_rows()):
		region = d.get_raw_value(i, 'BioPhyReg' )
		regionCats.append(regionDict[region])
	d.add_column("regionCat", numpy.matrix(regionCats).T)
	
	# categorize by month of arrival
	arrivalMonths = []
	for i in range(d.get_raw_num_rows()):
		day = d.get_value(i, 'DOY')
		if day < 60:
			arrivalMonths.append(0)
		elif day < 91:
			arrivalMonths.append(1)
		elif day < 121:
			arrivalMonths.append(2)
		else:
			arrivalMonths.append(3)
	d.add_column("arrivalCat", numpy.matrix(arrivalMonths).T)
	
	# delete rows missing data
	todelete = numpy.array([], dtype=numpy.int64)
	for i in range(d.get_raw_num_rows()):
		if -9999 in d.get_row(i):
			todelete = numpy.append(todelete, i)
	for r in range(todelete.shape[0]):
		d.delete_row(todelete[r])
		todelete = todelete-1
		
	d.write("ArrivalDates")