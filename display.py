# Martin Deutsch 
# Project 8
# Skeleton written by Bruce Maxwell
# Modified by Stephanie Taylor
#
# CS 251
# Spring 2017

import Tkinter as tk
import tkFont as tkf
import tkFileDialog
import numpy as np
import scipy.stats
import math

import view
import data
import analysis

# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):
        
        # create view object and axes
        self.view = view.View()
        self.axes = np.matrix([ [ 0., 0., 0., 1. ],
                                [ 1., 0., 0., 1. ],
                                [ 0., 0., 0., 1. ],
                                [ 0., 1., 0., 1. ],
                                [ 0., 0., 0., 1. ],
                                [ 0., 0., 1., 1. ] ])
        self.lines = []
        self.labels = []
        
        self.baseClick1 = None # keeps track of mouse button1 movement
        self.baseClick2 = None # keeps track of mouse button2 movement
        self.baseClick3 = None # keeps track of mouse button3 movement
        
        self.data = None # holds data object
        self.objects = [] # list of data objects that will be drawn in the canvas
        self.dataMatrix = np.matrix([]) # holds data points in matrix form
        
        self.sizeMatrix = np.matrix([]) # holds raw size data
        self.sizes = [] # holds the list of sizes for the data points
        
        self.colorMatrix = np.matrix([]) # holds the raw color data
        self.colors = [] # holds the list of colors for the data points
        
        self.regressionObjects = [] # holds the linear regression lines
        self.regressionMatrix = None # holds the normalized regression matrix
        
        self.pcaObjects = [] # holds the previous PCA data objects
        
        self.clusterData = None # holds the clustering data object
        
        # create a tk object, which is the root window
        self.root = tk.Tk()
        
        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("MARVIN")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1600, 900 )

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the Canvas
        self.buildCanvas()
        
        # build the axes
        self.buildAxes()

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print self.canvas.winfo_geometry()

        # set up the key bindings
        self.setBindings()

    def buildMenus(self):
        
        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu( menu )
        menu.add_cascade( label = "File", menu = filemenu )
        menulist.append(filemenu)

        # create a command menu
        cmdmenu = tk.Menu( menu )
        menu.add_cascade( label = "Command", menu = cmdmenu )
        menulist.append(cmdmenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [ [ '-', 'Open \xE2\x8C\x98-O', 'Quit  \xE2\x8C\x98-Q' ],
                     [ 'Linear Regression \xE2\x8C\x98-R', 'PCA \xE2\x8C\x98-P',
                        'Clustering \xE2\x8C\x98-C'] ]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [ [None, self.handleOpen, self.handleQuit],
                    [self.handleLinearRegression, self.handlePCA, self.handleClustering] ]
        
        # build the menu elements and callbacks
        for i in range( len( menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return
   
   # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        rightcntlframe = tk.Frame(self.root)
        rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )
        
        # make a button in the frame
        # and tell it to call the handleResetButton method when it is pressed.
        resetButton = tk.Button( rightcntlframe, text="Reset Axes", 
                               command=self.handleResetButton )
        resetButton.pack(side=tk.TOP)  # default side is top
        
        # make a button in the frame
        # and tell it to call the handlePlotData method when it is pressed.
        dataButton = tk.Button( rightcntlframe, text="Add data", 
                               command=self.handlePlotData )
        dataButton.pack(side=tk.TOP)  # default side is top
        
        # add label for data shape menu
        menuLabel = tk.Label( rightcntlframe, text="Data point shape" )
        menuLabel.pack( side=tk.TOP, pady=5 )
        # make a menu to allow the user to select the shape of the data points
        self.dataShape = tk.StringVar( self.root )
        self.dataShape.set("point")
        dataMenu = tk.OptionMenu( rightcntlframe, self.dataShape, 
                                        "point", "square", "circle" )
        dataMenu.pack(side=tk.TOP)
        
        # make a menu to hold the PCA analyses
        pcaLabel = tk.Label( rightcntlframe, text="PCAs run")
        pcaLabel.pack( side=tk.TOP, pady=5)
        self.pcaMenu = tk.Listbox( rightcntlframe, selectmode=tk.SINGLE, height=5 )
        self.pcaMenu.pack(side=tk.TOP)
        # make buttons to run or delete previous analyses
        buttonframe = tk.Frame(rightcntlframe)
        buttonframe.pack(side=tk.TOP, padx=2, pady=2)
        deleteButton = tk.Button(buttonframe, text="Delete",command=self.deletePCA)
        runButton = tk.Button(buttonframe, text="Run", command=self.runPCA)
        deleteButton.pack(side=tk.LEFT)
        runButton.pack(side=tk.LEFT)
        
        # make a button in the frame
        # and tell it to call the handlePlotData method when it is pressed.
        clusterButton = tk.Button( rightcntlframe, text="Show clusters", 
                               command=self.drawClustering )
        clusterButton.pack(side=tk.TOP)  # default side is top
        
        # add a legend
        self.legend = tk.Label( rightcntlframe )
        self.legend.pack( side=tk.TOP, pady=5 )
        self.legendCanvas = tk.Canvas( rightcntlframe, width=100 )
        self.legendCanvas.pack( side=tk.TOP, pady=5 )
        
        return
    
    # create and draw axes and data
    def buildAxes(self):
        vtm = self.view.build()
        
        # create axes and labels
        pts = (vtm * self.axes.T).T
        
        x = self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1])
        self.lines.append(x)
        xLabel = self.canvas.create_text(pts[1, 0]+10, pts[1, 1], text="x")
        self.labels = [xLabel]
        
        y = self.canvas.create_line(pts[2, 0], pts[2, 1], pts[3, 0], pts[3, 1])
        self.lines.append(y)
        yLabel = self.canvas.create_text(pts[3, 0]+10, pts[3, 1], text="y")
        self.labels.append(yLabel)
        
        z = self.canvas.create_line(pts[4, 0], pts[4, 1], pts[5, 0], pts[5, 1])
        self.lines.append(z)
        zLabel = self.canvas.create_text(pts[5, 0]+10, pts[5, 1], text="z")
        self.labels.append(zLabel)
    
    # build data matrix
    def buildPoints(self, cols):
        self.clear()

        # build data point matrix
        self.dataMatrix = analysis.normalize_columns_separately(self.data, cols)
        if len(cols) == 2:
            zeros = np.zeros(self.data.get_raw_num_rows())
            self.dataMatrix = np.hstack( (self.dataMatrix, np.matrix(zeros).T) )
        ones = np.ones(self.data.get_raw_num_rows())
        self.dataMatrix = np.hstack( (self.dataMatrix, np.matrix(ones).T) )

        # add to view screen
        vtm = self.view.build()
        pts = (vtm * self.dataMatrix.T).T
        shape = self.dataShape.get()
        for i in range(pts.shape[0]):
            if shape == "circle":
                pt = self.canvas.create_oval(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], outline=self.colors[i])
            elif shape == "square":
                pt = self.canvas.create_rectangle(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], fill=self.colors[i], outline='')
            else:
                pt = self.canvas.create_oval(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], fill=self.colors[i], outline='')
            self.objects.append(pt)
    
    # defines a color scheme of easily differentiated colors for up to 10 clusters
    def preselectColors(self, colorMatrix):
        colors = colorMatrix.T.tolist()[0]
        for i in range(colorMatrix.shape[0]):
            if colorMatrix[i, 0] == 0:
                rgb = (255, 0, 0) # red
            elif colorMatrix[i, 0] == 1:
                rgb = (0, 0, 255) # blue
            elif colorMatrix[i, 0] == 2:
                rgb = (0, 255, 0) # green
            elif colorMatrix[i, 0] == 3:
                rgb = (255, 235, 0) # yellow
            elif colorMatrix[i, 0] == 4:
                rgb = (255, 0, 255) # purple
            elif colorMatrix[i, 0] == 5:
                rgb = (0, 255, 255) # pink
            elif colorMatrix[i, 0] == 6:
                rgb = (255, 140, 0) # orange
            elif colorMatrix[i, 0] == 7:
                rgb = (0, 100, 0) # dark green
            elif colorMatrix[i, 0] == 8:
                rgb = (0, 0, 80) # navy
            elif colorMatrix[i, 0] == 9:
                rgb = (137, 137, 137) # grey
            else:
                rgb = (0, 0, 0) # black
            colors[i] = ('#%02x%02x%02x' % rgb)
        return colors
        
    # update screen appearance of axes 
    def updateAxes(self):
        vtm = self.view.build()
        
        # multiply the axis endpoints by the VTM
        pts = (vtm * self.axes.T).T
        
        # update coordinates of the line objects and labels
        for i in range( len(self.lines) ):
            self.canvas.coords(self.lines[i], pts[i*2, 0], pts[i*2, 1], 
                                            pts[i*2+1, 0], pts[i*2+1, 1])
            if i < 3:
                self.canvas.coords(self.labels[i], pts[i*2+1, 0]+10, pts[i*2+1, 1])
                
    # update screen appearance of points
    def updatePoints(self):
        if (len(self.objects) == 0):
            return
        
        vtm = self.view.build()
        # multiply data matrix by the VTM
        pts = (vtm * self.dataMatrix.T).T
        for i in range( len(self.objects) ):
            self.canvas.coords(self.objects[i], pts[i, 0]-self.sizes[i], 
                            pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                            pts[i, 1]+self.sizes[i])
    
    # update positions of data points and fit line in regression analysis
    def updateFits(self):
        if (len(self.regressionObjects) == 0):
            return
        
        vtm = self.view.build()
        # multiply endpoint matrix by the VTM
        pts = (vtm * self.regressionMatrix.T).T
        for i in range( len(self.regressionObjects) ):
            self.canvas.coords(self.regressionObjects[i], pts[i*2, 0], 
                                 pts[i*2, 1], pts[i*2+1, 0], pts[i*2+1, 1])
            self.canvas.coords(self.labels[i+3], pts[i*2+1, 0]+120, pts[i*2+1, 1]+20)

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
        self.canvas.bind( '<Shift-Command-Button-1>', self.handleMouseButton3)
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Shift-Command-B1-Motion>', self.handleMouseButton3Motion)

        # bind command sequences to the root window
        self.root.bind( '<Command-q>', self.handleQuit )
        self.root.bind( '<Command-o>', self.handleOpen )
        self.root.bind( '<Command-r>', self.handleLinearRegression )
        self.root.bind( '<Command-p>', self.handlePCA )
        self.root.bind( '<Command-c>', self.handleClustering)

    # allow the user to choose a data file
    def handleOpen(self, event=None):
        self.fn = tkFileDialog.askopenfilename( parent=self.root,
                title='Choose a data file', initialdir='.' )
        self.data = data.Data(self.fn)
    
    # terminate root window
    def handleQuit(self, event=None):
        print 'Terminating'
        self.root.destroy()
        
    # This is called if reset button is pressed
    def handleResetButton(self):
        self.reset()
        
    # This is called if the add data button is pressed
    def handlePlotData(self):
        self.headers = self.handleChooseAxes()
        self.legendCanvas.delete("all")
        self.legend.config(text="")
        self.canvas.itemconfig(self.labels[0], text="x")
        self.canvas.itemconfig(self.labels[1], text="y")
        self.canvas.itemconfig(self.labels[2], text="z")
        
        if self.headers:
            # handle axis data
            cols = [self.headers["x"], self.headers["y"]]
            if "z" in self.headers:
                cols.append(self.headers["z"])
            
            # handle color data
            if "color" in self.headers:
                self.colorMatrix = self.data.get_data([self.headers["color"]])
                if self.pre == 1:
                    self.colors = self.preselectColors(self.colorMatrix)
                else:
                    # normalize column with mean and standard deviation
                    mean = np.mean(self.colorMatrix, axis=0)
                    std = np.std(self.colorMatrix, axis=0)
                    tmp = self.colorMatrix - mean
                    if std == 0:
                        color = np.matrix( np.zeros((self.colorMatrix.shape[0],
                                                self.colorMatrix.shape[1]) ))
                    else:
                        color = tmp / np.matrix(std, dtype=float)
                    color = (color+2.5)/5
                    # create color list
                    self.colors = color.T.tolist()[0]
                    for i in range(self.data.get_raw_num_rows()):
                        if color[i, 0] < 0:
                            color[i, 0] = 0
                        if color[i, 0] > 1:
                            color[i, 0] = 1
                        rgb = ( 0, color[i, 0]*255, (1-color[i, 0])*255 )
                        self.colors[i] = ('#%02x%02x%02x' % rgb)

                # create color legend
                self.legend.config(text="Legend")
                self.legendCanvas.create_rectangle(20, 14, 26, 20, 
                                        fill='#0000ff', outline='')
                self.legendCanvas.create_rectangle(40, 14, 46, 20, 
                                        fill='#0055be', outline='')
                self.legendCanvas.create_rectangle(60, 14, 66, 20, 
                                        fill='#00be55', outline='')
                self.legendCanvas.create_rectangle(80, 14, 86, 20, 
                                        fill='#00ff00', outline='')
                self.legendCanvas.create_text( 20, 30, 
                            text=int(np.min(self.colorMatrix, axis=0)[0, 0]) )
                self.legendCanvas.create_text(90, 30,
                            text=int(np.max(self.colorMatrix, axis=0)[0, 0]) )
            else:
                self.colors = ['#000000']*self.data.get_raw_num_rows()
                
            # handle size data
            if "size" in self.headers:
                self.sizeMatrix = self.data.get_data([self.headers["size"]])
                # normalize column
                min = np.min(self.sizeMatrix, axis=0)
                max = np.max(self.sizeMatrix, axis=0)
                tmp = self.sizeMatrix - min
                dataRange = max - min
                dataRange[dataRange == 0] = 1
                size = tmp / np.matrix(dataRange, dtype=float)
                # create size list
                self.sizes = size.T.tolist()[0]
                for i in range(len(self.sizes)):
                    self.sizes[i] = int(math.sqrt(self.sizes[i])*3+1)
                    
                # create size legend
                self.legend.config(text="Legend")
                self.legendCanvas.create_rectangle(20, 78, 22, 80, 
                                        fill='#000000', outline='')
                self.legendCanvas.create_rectangle(40, 76, 44, 80, 
                                        fill='#000000', outline='')
                self.legendCanvas.create_rectangle(60, 74, 66, 80, 
                                        fill='#000000', outline='')
                self.legendCanvas.create_rectangle(80, 72, 88, 80, 
                                        fill='#000000', outline='')
                self.legendCanvas.create_text(20, 90, text=int(min[0, 0]))
                self.legendCanvas.create_text(90, 90, text=int(max[0, 0]))
            else:
                self.sizes = [2]*self.data.get_raw_num_rows()
                
            self.buildPoints(cols)
    
    # This lets the user choose which axes to plot
    def handleChooseAxes(self):
        if self.data == None:
            print "Choose input file"
            return
        
        dialog = DataDialog(self.root, self.data.get_headers(), "Choose columns")
        # Keep opening dialog box until x and y axes are chosen
        if dialog.x == "NaN" or dialog.y=="NaN":
            print "You must choose columns for the x and y axes!"
            return self.handleChooseAxes()

        if dialog.x == "":
            return {}
        headers = {"x": dialog.x, "y": dialog.y}
        if dialog.z != "":
            headers["z"] = dialog.z
        if dialog.size != "":
            headers["size"] = dialog.size
        if dialog.color != "":
            headers["color"] = dialog.color
        self.pre = dialog.pre.get()
        return headers

    # This is called if mouse-button-1 is clicked
    def handleMouseButton1(self, event):
        self.baseClick1 = (event.x, event.y)
        self.mouse1motion = self.baseClick1
        
    # This is called if mouse-button-2 or
    # control-mouse-button-1 is clicked
    def handleMouseButton2(self, event):
        self.baseClick2 = (event.x, event.y)
        self.originalView = self.view.clone()
    
    # This is called if shift-cmd-mouse-button-1 is clicked
    def handleMouseButton3(self, event):
        self.baseClick3 = (event.x, event.y)
        self.extent = self.view.clone().extent
            
    # This is called if the first mouse button is being moved
    def handleMouseButton1Motion(self, event):
        # calculate the difference
        diff = ( event.x - self.mouse1motion[0], event.y - self.mouse1motion[1] )
        self.mouse1motion = ( event.x, event.y )
        
        # calculate scale factor
        diff = [float(diff[0]) / self.view.screen[0], 
                        float(diff[1]) / self.view.screen[1]]
        delta0 = diff[0] * self.view.extent[0]
        delta1 = diff[1] * self.view.extent[1]
        
        # update VRP
        self.view.vrp += delta0*self.view.u + delta1*self.view.vup
        self.updateAxes()
        self.updatePoints()
        self.updateFits()
            
    # This is called if the second button of a real mouse has been pressed
    # and the mouse is moving, or if the control key is held down while
    # a person moves their finger on the track pad.
    def handleMouseButton2Motion(self, event):
        delta0 = float(self.baseClick2[0] - event.x) / 400 * math.pi
        delta1 = float(self.baseClick2[1] - event.y) / 400 * math.pi
        self.view = self.originalView.clone()
        self.view.rotateVRC(delta0, -delta1)
        self.updateAxes()
        self.updatePoints()
        self.updateFits()
 
    # This is called if shift-cmd-mouse-button-1 is held while
    # the mouse is moving
    def handleMouseButton3Motion(self, event):
        # calculate scale factor
        diff = self.baseClick3[1] - event.y
        factor = float(diff) / 400
        if factor < 0: # zoom out
            factor = abs(factor) + 1
        else: # zoom in
            factor = 1 / (1 + factor)
        # update extent
        self.view.extent = [self.extent[0]*factor, self.extent[1]*factor, 
                                    self.extent[2]*factor]
        self.updateAxes()
        self.updatePoints()
        self.updateFits()
        
    # This is called if the user executes the linear regression command
    def handleLinearRegression(self, event=None):
    	if self.data == None:
            print "Choose input file"
            return
        
        # Create selection dialog box
        dialog = RegressionDialog(self.root, self.data.get_headers(), 
                                        "Select variables to fit")
        # Terminate if cancelled
        if dialog.indx == '':
            return
       
        # Keep opening dialog box until proper variables are chosen
        if dialog.indx == 'NaN' or dialog.dep == 'NaN' or dialog.indx == dialog.indz:
            print "Choose seperate dependent variables and an independent variable"
            self.handleLinearRegression()
            return
        
        self.clear()
        self.reset()
        self.buildLinearRegression(dialog.indx, dialog.indz, dialog.dep, 
                                        dialog.out.get(), dialog.filename.get())
    
    # This is called if the user executes the PCA command
    def handlePCA(self, event=None):
        if self.data == None:
            print "Choose input file"
            return
        
        # Create selection dialog box
        dialog = PCADialog(self.root, self.data.get_headers(), 
                                "Select columns for PCA analysis")
        # Terminate if cancelled
        if len(dialog.cols) == 0:
            return
        elif len(dialog.cols) == 1:
            print "Select at least 2 columns for PCA analysis"
            return
        
        self.clear()
        self.reset()
        self.buildPCA(dialog.cols, dialog.name.get())
        
    # delete the selected PCA analysis from the listbox
    def deletePCA(self):
        if self.pcaMenu.curselection():
            del self.pcaObjects[ self.pcaMenu.curselection()[0] ]
            self.pcaMenu.delete(tk.ACTIVE)
        else:
            print "Select a PCA analysis to delete"
        
    # run the selected PCA analysis
    def runPCA(self):
        if self.pcaMenu.curselection():
            self.clear()
            self.reset()
            self.drawPCA(self.pcaObjects[ self.pcaMenu.curselection()[0] ])
        else:
            print "Select a PCA analysis to run"
            
    # This is called if the user executes the clustering command
    def handleClustering(self, event=None):
        if self.data == None:
            print "Choose input file"
            return
        
        # Create selection dialog box
        dialog = clusterDialog(self.root, self.data.get_headers(), 
                                    "Select columns for clustering analysis")
        # Terminate if cancelled
        if len(dialog.cols) == 0:
            return
        elif len(dialog.cols) == 1:
            print "Select at least 2 columns for clustering analysis"
            return
        
        try:
            n = int(dialog.num.get())
        except ValueError:
            print "Please enter an integer for the number of clusters"
            return
        
        if n == 0:
            print "Please enter a positive integer for the number of clusters"
            return
    
        self.means, ids, self.errors = analysis.kmeans(self.data, 
                                                dialog.cols, n, dialog.metric)
        
        self.clusterData = data.ClusterData(self.data.get_headers(), 
                                self.data.get_data(self.data.get_headers()))
        self.clusterData.add_column(str(n)+"clusterIds", ids)
        print "Press 'Show clusters' and select the clusterIds column to view clustering"
        
    # display the clustering data on the view screen
    def drawClustering(self):
        if self.clusterData == None:
            print "Run a cluster analysis"
            return
        
        dialog = DataDialog(self.root, self.clusterData.get_headers(), "Choose columns")
        if dialog.x == "NaN" or dialog.y=="NaN":
            print "You must choose columns for the x and y axes"
            return
        
        self.clear()
        self.reset()
            
        if (dialog.z != ""):
            matrix = analysis.normalize_columns_separately(self.clusterData, 
                                    [dialog.x, dialog.y, dialog.z])
        else:
            matrix = analysis.normalize_columns_separately(self.clusterData, 
                                    [dialog.x, dialog.y])
            zeros = np.zeros(self.data.get_raw_num_rows())
            matrix = np.hstack( (matrix, np.matrix(zeros).T) )
            
        ones = np.ones(self.data.get_raw_num_rows())
        self.dataMatrix = np.hstack( (matrix, np.matrix(ones).T) )
        # calculate view coordinates
        vtm = self.view.build()
        pts = (vtm * self.dataMatrix.T).T
        
        if (dialog.size != ''):
            size = analysis.normalize_columns_separately(self.clusterData, [dialog.size])
            self.sizes = size.T.tolist()[0]
            for i in range(len(self.sizes)):
                    self.sizes[i] = int(math.sqrt(self.sizes[i])*3+1)
        else:
            self.sizes = [2]*self.clusterData.get_raw_num_rows()
        
        # handle color data
        if dialog.color != "":
            self.colorMatrix = self.clusterData.get_data([dialog.color])
            if dialog.pre.get() == 1:
                self.colors = self.preselectColors(self.colorMatrix)
            else:
                # normalize column with mean and standard deviation
                mean = np.mean(self.colorMatrix, axis=0)
                std = np.std(self.colorMatrix, axis=0)
                tmp = self.colorMatrix - mean
                if std == 0:
                    color = np.matrix( np.zeros((self.colorMatrix.shape[0],
                                            self.colorMatrix.shape[1]) ))
                else:
                    color = tmp / np.matrix(std, dtype=float)
                color = (color+2.5)/5
                # create color list
                self.colors = color.T.tolist()[0]
                for i in range(self.data.get_raw_num_rows()):
                    if color[i, 0] < 0:
                        color[i, 0] = 0
                    if color[i, 0] > 1:
                        color[i, 0] = 1
                    rgb = ( 0, color[i, 0]*255, (1-color[i, 0])*255 )
                    self.colors[i] = ('#%02x%02x%02x' % rgb)
        else:
            self.colors = ['#000000']*self.data.get_raw_num_rows()
        
        for i in range(len(pts)):
            pt = self.canvas.create_oval(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], fill=self.colors[i], outline='')
            self.objects.append(pt)
        
    # This clears the canvas of data points and regression models
    def clear(self):
        for obj in self.objects:
            self.canvas.delete(obj)
        self.objects = []
        for obj in self.regressionObjects:
            self.canvas.delete(obj)
        self.regressionObjects = []
        if (len(self.labels) == 4):
            self.canvas.delete(self.labels[3])
            del self.labels[3]
        
    # This resets the axes to their default position
    def reset(self):
        self.view.reset()
        self.updateAxes()
        self.updatePoints()
        self.updateFits()
        
    # This builds the regression model
    def buildLinearRegression(self, indx, indz, dep, export, filename):
        if (indz != ''):
            matrix = analysis.normalize_columns_separately(self.data, [indx, dep, indz])
        else:
            matrix = analysis.normalize_columns_separately(self.data, [indx, dep])
            zeros = np.zeros(self.data.get_raw_num_rows())
            matrix = np.hstack( (matrix, np.matrix(zeros).T) )
            
        ones = np.ones(self.data.get_raw_num_rows())
        self.dataMatrix = np.hstack( (matrix, np.matrix(ones).T) )
        # calculate view coordinates
        vtm = self.view.build()
        pts = (vtm * self.dataMatrix.T).T
        
        # use points with default size and color
        self.sizes = [2]*self.data.get_raw_num_rows()
        self.colors = ['#000000']*self.data.get_raw_num_rows()
        for i in range(len(pts)):
            pt = self.canvas.create_oval(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], fill=self.colors[i], outline='')
            self.objects.append(pt)
        
        # calculate single variable linear regression
        if (indz == ''):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                                        self.data.get_data([indx, dep]))
            ranges = analysis.data_range(self.data, [indx, dep])
            end1y = ((ranges[0][0]*slope+intercept)-ranges[1][0])/(ranges[1][1]-ranges[1][0])
            end2y = ((ranges[0][1]*slope+intercept)-ranges[1][0])/(ranges[1][1]-ranges[1][0])
            self.regressionMatrix = np.matrix([ [0.0, end1y, 0.0, 1.0],
                                                [1.0, end2y, 0.0, 1.0] ])
                                                
            eqn = "y = %.3fx + %.3f \nR = %.3f" % (slope, intercept, r_value)
            data = "p = %.3f \nStandard error = %.3f" % (p_value, std_err)
            out = eqn + "\n" + data
            
        # calculate muliple variable linear regression
        else:
            b, sse, r2, t, p = analysis.linear_regression(self.data, [indx, indz], dep)
            ranges = analysis.data_range(self.data, [indx, indz, dep])
            end1y = ranges[0][0]*b[0] + ranges[1][0]*b[1] + b[2]
            end1y = (end1y - ranges[2][0])/(ranges[2][1] - ranges[2][0])
            end2y = ranges[0][1]*b[0] + ranges[1][1]*b[1] + b[2]
            end2y = (end2y - ranges[2][0])/(ranges[2][1] - ranges[2][0])
            self.regressionMatrix = np.matrix([ [0.0, end1y, 0.0, 1.0],
                                                [1.0, end2y, 1.0, 1.0] ])
                                                
            eqn =  "y = %.3fx + %.3fz + %.3f \nR^2 = %.3f" % (b[0], b[1], b[2], r2)
            sse_data = "Sum-squared error = %.3f" % (sse)
            p_data = "p = [%.3f, %.3f, %.3f]" % (p[0, 0], p[0, 1], p[0, 2])
            t_data = "t-statistic = [%.3f, %.3f, %.3f]" % (t[0, 0], t[0, 1], t[0, 2])
            out = eqn + "\n" + sse_data + "\n" + p_data + "\n" + t_data
            
        # display regression onscreen
        self.canvas.itemconfig(self.labels[0], text="x")
        self.canvas.itemconfig(self.labels[1], text="y")
        self.canvas.itemconfig(self.labels[2], text="z")
        endpts = (vtm * self.regressionMatrix.T).T
        l = self.canvas.create_line(endpts[0, 0], endpts[0, 1], endpts[1, 0], 
                                        endpts[1, 1], fill="red")
        self.regressionObjects.append(l)
        regLabel = self.canvas.create_text(endpts[1, 0]+120, endpts[1, 1]+20, text=eqn)
        self.labels.append(regLabel)
        title = "Linear regression for " + str(self.fn)
    
        # write linear regression function to file
        if (export == 1):
            file = open(filename + ".txt", 'w')
            file.write(title + "\n" + out)
            file.close()
            
    # This executes the PCA
    def buildPCA(self, cols, name):
        pcad = analysis.pca(self.data, cols)
        self.pcaObjects.append(pcad)
        self.pcaMenu.insert(tk.END, name)
        self.drawPCA(pcad)
        
    # This draws the PCA analysis on the axes
    def drawPCA(self, pcad):
        dialog = eigenDialog(self.root, pcad, "Eigenvectors")
        self.canvas.itemconfig(self.labels[0], text=dialog.x.get())
        self.canvas.itemconfig(self.labels[1], text=dialog.y.get())
        self.canvas.itemconfig(self.labels[2], text=dialog.z.get())
        cols = [dialog.x.get(), dialog.y.get()]
        if dialog.z.get():
            cols.append(dialog.z.get())
        
        # create data matrix
        matrix = pcad.get_data(cols)
        if matrix.shape[1] == 2:
            zeros = np.zeros(pcad.get_raw_num_rows())
            matrix = np.hstack( (matrix, np.matrix(zeros).T) )
        ones = np.ones(pcad.get_raw_num_rows())
        self.dataMatrix = np.hstack( (matrix, np.matrix(ones).T) )
        
        # calculate view coordinates
        vtm = self.view.build()
        pts = (vtm * self.dataMatrix.T).T

        # get sizes
        if dialog.size.get():
            self.sizeMatrix = pcad.get_data([dialog.size.get()])
            # normalize column
            min = np.min(self.sizeMatrix, axis=0)
            max = np.max(self.sizeMatrix, axis=0)
            tmp = self.sizeMatrix - min
            dataRange = max - min
            dataRange[dataRange == 0] = 1
            size = tmp / np.matrix(dataRange, dtype=float)
            # create size list
            self.sizes = size.T.tolist()[0]
            for i in range(len(self.sizes)):
                self.sizes[i] = int(math.sqrt(self.sizes[i])*3+1)
        else:
            self.sizes = [2]*pcad.get_raw_num_rows()
        
        # get colors  
        if dialog.color.get():
            self.colorMatrix = pcad.get_data([dialog.color.get()])
            # normalize column with mean and standard deviation
            mean = np.mean(self.colorMatrix, axis=0)
            std = np.std(self.colorMatrix, axis=0)
            tmp = self.colorMatrix - mean
            color = tmp / np.matrix(std, dtype=float)
            color = (color+2.5)/5
            # create color list
            self.colors = color.T.tolist()[0]
            for i in range(self.data.get_raw_num_rows()):
                if color[i, 0] < 0:
                    color[i, 0] = 0
                if color[i, 0] > 1:
                    color[i, 0] = 1
                rgb = ( 0, color[i, 0]*255, (1-color[i, 0])*255 )
                self.colors[i] = ('#%02x%02x%02x' % rgb)
        else:
            self.colors = ['#000000']*pcad.get_raw_num_rows()
       
        # build points
        for i in range(len(pts)):
            pt = self.canvas.create_oval(pts[i, 0]-self.sizes[i], 
                        pts[i, 1]-self.sizes[i], pts[i, 0]+self.sizes[i], 
                        pts[i, 1]+self.sizes[i], fill=self.colors[i], outline='')
            self.objects.append(pt)
    
    # run main loop until quit
    def main(self):
        print "Welcome to MARVIN!"
        print "(Multiple Analysis Rendering Visualization for Interesting Numbers)"
        print "Open a data file to begin analysis."
        self.root.mainloop()

# Support class for creating dialog boxes
class Dialog(tk.Toplevel):

    def __init__(self, parent, title = None):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):
        
        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

# Creates dialog box for entering columns for axes
class DataDialog(Dialog):
    
    def __init__(self, parent, colHeaders, title = None):
        self.headers = colHeaders
        self.pre = tk.IntVar()
        Dialog.__init__(self, parent, title)
        
    # Create labels and listboxes for axes
    def body(self, master):
        tk.Label(master, text="x axis:").grid(row=0)
        tk.Label(master, text="y axis:").grid(row=1)
        tk.Label(master, text="z axis:").grid(row=2)
        tk.Label(master, text="color:").grid(row=3)
        tk.Label(master, text="size:").grid(row=6)

        self.lbx = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        
        self.lby = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        
        self.lbz = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
                                        
        self.lbcolor = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
                                        
        self.lbsize = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        self.preselect = tk.Checkbutton(master, var=self.pre, 
                                        text="Use preselected color scheme")
                                        
        for header in self.headers:
            self.lbx.insert(tk.END, header)
            self.lby.insert(tk.END, header)
            self.lbz.insert(tk.END, header)
            self.lbcolor.insert(tk.END, header)
            self.lbsize.insert(tk.END, header)
        
        self.lbx.grid(row=0, column=1)
        self.lby.grid(row=1, column=1)
        self.lbz.grid(row=2, column=1)
        self.lbcolor.grid(row=3, column=1)
        self.preselect.grid(row=5, column=1)
        self.lbsize.grid(row=6, column=1)
        
        self.x = ""
        self.y = ""
        self.z = ""
        self.color = ""
        self.size = ""
        return self.lbx # initial focus

    # store selected column headers in fields
    def apply(self):
        if self.lbx.curselection():
            self.x = self.headers[self.lbx.curselection()[0]]
        else:
            self.x = "NaN"
        if self.lby.curselection():
            self.y = self.headers[self.lby.curselection()[0]]
        else:
            self.y = "NaN"
        if self.lbz.curselection():
            self.z = self.headers[self.lbz.curselection()[0]]
        if self.lbcolor.curselection():
            self.color = self.headers[self.lbcolor.curselection()[0]]
        if self.lbsize.curselection():
            self.size = self.headers[self.lbsize.curselection()[0]]
             
# Creates dialog box to select columns for regression analysis
class RegressionDialog(Dialog):
    
    def __init__(self, parent, colHeaders, title = None):
        self.headers = colHeaders
        self.out = tk.IntVar()
        self.filename = tk.StringVar()
        Dialog.__init__(self, parent, title)
        
    # Create labels and listboxes for axes
    def body(self, master):
        tk.Label(master, text="Independent variable x:").grid(row=0)
        tk.Label(master, text="Independent variable z (Optional):").grid(row=1)
        tk.Label(master, text="Dependent variable y:").grid(row=2)

        self.lbx = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        self.lbz = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        self.lby = tk.Listbox(master, selectmode=tk.SINGLE, 
                                        exportselection=0, height=5)
        self.c = tk.Checkbutton(master, text="Save to file?", variable=self.out)
        self.e = tk.Entry(master, textvariable=self.filename)
        self.e.insert(0, "filename")
        
        for header in self.headers:
            self.lbx.insert(tk.END, header)
            self.lbz.insert(tk.END, header)
            self.lby.insert(tk.END, header)
        
        self.lbx.grid(row=0, column=1)
        self.lbz.grid(row=1, column=1)
        self.lby.grid(row=2, column=1)
        self.c.grid(row=3, column=0)
        self.e.grid(row=3, column=1)
        
        self.indx = ''
        self.indz = ''
        self.dep = ''
        
        return self.lbx # initial focus

    # return selected columns
    def apply(self):
        if self.lbx.curselection():
            self.indx = self.headers[ self.lbx.curselection()[0] ]
        else:
            self.indx = "NaN"
        if self.lbz.curselection():
            self.indz = self.headers[ self.lbz.curselection()[0] ]
        if self.lby.curselection():
            self.dep = self.headers[ self.lby.curselection()[0] ]
        else:
            self.dep = "NaN"
        
# Creates dialog box to select columns for PCA
class PCADialog(Dialog):
    
    def __init__(self, parent, colHeaders, title = None):
        self.headers = colHeaders
        self.name = tk.StringVar()
        Dialog.__init__(self, parent, title)
        
    # Create labels and listboxes for axes
    def body(self, master):
        tk.Label(master, text="Select Columns:").grid(row=0)
        tk.Label(master, text="Analysis Name:").grid(row=1, column=0)

        self.lb = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
        self.e = tk.Entry(master, textvariable=self.name)
        self.e.insert(0, "PCA")
        
        for header in self.headers:
            self.lb.insert(tk.END, header)
        
        self.lb.grid(row=0, column=1)
        self.e.grid(row=1, column=1)
        
        self.cols = []
        
        return self.lb # initial focus

    # return selected columns
    def apply(self):
        if self.lb.curselection():
            for i in self.lb.curselection():
                self.cols.append(self.headers[i])
            
# Creates dialog box to select columns for PCA
class eigenDialog(Dialog):
        
    def __init__(self, parent, pcad, title = None):
        self.pca = pcad
        self.x = tk.StringVar()
        self.x.set("e0")
        self.y = tk.StringVar()
        self.y.set("e1")
        self.z = tk.StringVar()
        self.color = tk.StringVar()
        self.size = tk.StringVar()
        Dialog.__init__(self, parent, title)
        
    # Create grid showing projected data
    def body(self, master):
        tk.Label(master, text="E-vec").grid(row=0, column=0)
        evecs = self.pca.get_raw_headers()
        for i in range( len(evecs) ):
            tk.Label(master, text=evecs[i]).grid(row=i+1, column=0)
        
        tk.Label(master, text="E-val").grid(row=0, column=1)
        evals = self.pca.get_eigenvalues()
        for i in range( evals.shape[1] ):
            tk.Label( master, text=('%.4f' % evals[0, i]) ).grid(row=i+1, column=1)
            
        tk.Label(master, text="Cumulative").grid(row=0, column=2)
        sum = np.sum(evals)
        for i in range( evals.shape[1] ):
            tk.Label( master, 
                text=('%.4f' % (np.sum(evals[0, 0:i+1])/sum)) ).grid(row=i+1, column=2)
      
        headers = self.pca.get_data_headers()
        data = self.pca.get_eigenvectors()
        for i in range( len(headers) ):
            tk.Label(master, text=headers[i]).grid(row=0, column=i+3)
            for n in range( len(evecs) ):
                tk.Label(master, 
                        text=('%.4f' % data[i, n])).grid(row=i+1, column=n+3)
        
        # allow the user to select which columns to plot
        tk.Label(master, text="").grid(row=len(headers)+1, column=0)
        tk.Label(master, text="x-axis").grid(row=len(headers)+2, column=0)
        tk.Label(master, text="y-axis").grid(row=len(headers)+2, column=1)
        tk.Label(master, text="z-axis").grid(row=len(headers)+2, column=2)
        tk.Label(master, text="Color").grid(row=len(headers)+2, column=3)
        tk.Label(master, text="Size").grid(row=len(headers)+2, column=4)
        x = apply(tk.OptionMenu, (master, self.x) + tuple(evecs))
        x.grid(row=len(headers)+3, column=0)
        y = apply(tk.OptionMenu, (master, self.y) + tuple(evecs))
        y.grid(row=len(headers)+3, column=1)
        z = apply(tk.OptionMenu, (master, self.z) + tuple(evecs))
        z.grid(row=len(headers)+3, column=2)
        color = apply(tk.OptionMenu, (master, self.color) + tuple(evecs))
        color.grid(row=len(headers)+3, column=3)
        size = apply(tk.OptionMenu, (master, self.size) + tuple(evecs))
        size.grid(row=len(headers)+3, column=4)

    # return selected columns
    def apply(self):
        return

# Creates dialog box to select columns for clustering
class clusterDialog(Dialog):
    
    def __init__(self, parent, colHeaders, title = None):
        self.headers = colHeaders
        self.num = tk.StringVar()
        self.metric = tk.StringVar()
        self.metric.set("L1 norm")
        Dialog.__init__(self, parent, title)
        
    # Create labels and listboxes for axes
    def body(self, master):
        tk.Label(master, text="Select Columns:").grid(row=0)
        tk.Label(master, text="Number of clusters").grid(row=1, column=0)
        tk.Label(master, text="Distance metric").grid(row=2, column=0)

        self.lb = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
        self.n = tk.Entry(master, textvariable=self.num)
        self.n.insert(0, "0")
        
        for header in self.headers:
            self.lb.insert(tk.END, header)
        
        self.lb.grid(row=0, column=1)
        self.n.grid(row=1, column=1)
        
        self.cols = []
        metricMenu = tk.OptionMenu( master, self.metric, 
                                    "L1 norm", "L2 norm", "L-inf norm" )
        metricMenu.grid(row=2, column=1)
        
        return self.lb # initial focus

    # return selected columns
    def apply(self):
        if self.lb.curselection():
            for i in self.lb.curselection():
                self.cols.append(self.headers[i])
                
# Run application
if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()