
import cv2
import sys
import numpy as np
import pygame
import matplotlib.pyplot as plt


from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QPoint
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QWidget, QInputDialog
from PyQt5.uic import loadUi
from collections import OrderedDict
from vectors import Point, Vector
from vpython import *
from PIL import Image




class GUI(QDialog):
    def __init__(self):
        super(GUI, self).__init__()
        loadUi(r"F:\Visual Studio 2019\FinalYearProject\FinalYearProject\GUI.ui", self)  # load the GUI file
        self.sourceImage = None  # set image to none
        self.textureImage = None
        self.dstImage = cv2.imread("blank_image.jpg", cv2.IMREAD_COLOR)
        self.outputImage = None
        self.gridImage = None
        self.meshgridCreated = False
        self.textureExtracted = False
        self.loadButton.clicked.connect(self.load_clicked)  # connect load button to loadClicked function
        self.line1button.clicked.connect(self.select_line1)
        self.line2button.clicked.connect(self.select_line2)
        self.line3button.clicked.connect(self.select_line3)
        self.midpointBtn.clicked.connect(self.select_midpoint)
        self.finishButton.clicked.connect(self.finish_line_selection)
        self.lowestPointBtn.clicked.connect(self.select_lowest_point)
        self.dimensionInputBtn.clicked.connect(self.room_dimensions)
        self.floorPercentageBtn.clicked.connect(self.ground_length_percentage)
        self.createMeshgridBtn.clicked.connect(self.implement_meshgrid)
        self.drawMeshgridBtn.clicked.connect(self.draw_meshgrid)
        self.createModelBtn.clicked.connect(self.create_model_3D)
        self.extractTextureBtn.clicked.connect(self.texture_extraction)
        self.mapMovementBtn.clicked.connect(self.movement_map)
        self.movementCoord1Btn.clicked.connect(self.select_mc1)
        self.movementCoord2Btn.clicked.connect(self.select_mc2)
        self.movementCoord3Btn.clicked.connect(self.select_mc3)
        self.movementCoord4Btn.clicked.connect(self.select_mc4)
        self.movementCoord5Btn.clicked.connect(self.select_mc5)
        self.finishButton3.clicked.connect(self.finish_mc_selection)
        self.ground = Ground()
        self.modelCreated = False
        self.scene = None
        self.mouse_selection = False # Mouse selection status for texture extraction
        self.texture_roi = [] #Region of interest for texture extraction

    ##__init__##


    ##Note: All pygtSlot functions are linked to GUI buttons

    #Load clicked retrieves the image and loads it
    @pyqtSlot()
    def load_clicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.jpg)")
        if fname:
            self.load_image(fname)

            #If the image is being loaded/reloaded and the model has not been created
            #set reset meshgrid to prevent order of operation error
            if self.modelCreated is False:
                self.meshgridCreated = False
                self.currentOperationTextEdit.clear()
                self.currentOperationTextEdit.append('1.) Define ground plane\n2.) Enter room dimensions and visible length percentage'
                                                     '\n3.) Create Meshgrid')

    ##load_clicked##


    #Accepts user input for room dimensions
    @pyqtSlot()
    def room_dimensions(self):
        width = QInputDialog.getInt(self, "Room Width", "Enter Width", 0, 0)
        length = QInputDialog.getInt(self, "Room Length", "Enter Length", 0, 0)

        #Extract width and length
        self.ground.width = width[0]
        self.ground.length = length[0]

        #Boundary checking
        if (width[0] > 1000 or length[0] > 1000 or width[0] < 100 or length[0] < 100):
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Dimensions are from 100 - 1000')
        else:
            #if ground is longer
            if(self.ground.length > self.ground.width):
                self.ground.isLonger = True
            #if ground is wider
            elif(self.ground.width > self.ground.length):
                self.ground.isWider = True


            #Boundary checking
            if self.ground.width <= 0 or self.ground.length <= 0:
                self.errorOutTextEdit.clear()
                self.errorOutTextEdit.append('Dimensions are from 100 - 1000')
            else:
                self.ground.dimDefined = True


    ##room_dimensions##


    #Accepts user input for the visible ground percentage
    @pyqtSlot()
    def ground_length_percentage(self):

        lengthPercentage = QInputDialog.getInt(self, "Length Percentage", "Enter Percentage", 0, 0)

        if lengthPercentage[0] < 40 or lengthPercentage[0] > 100:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Enter a valid percentage')
        else:
            self.errorOutTextEdit.clear()
            self.ground.plane.visLengthPercentage = float(lengthPercentage[0])

    ##ground_length_percentage##


    #Load image
    def load_image(self, fname):
        self.sourceImage = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.gridImage = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.textureImage = cv2.imread(fname, cv2.IMREAD_COLOR)

        #Checking the size of image
        if self.sourceImage.shape[0] < 861 or self.sourceImage.shape[1] < 1321:
            self.display_image("source", 1)
            self.errorOutTextEdit.clear()
        else:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Image is too big. Max size is 1321 x 861')

    ##load_image##


    #Display the image
    #The writing of this code was informed by openCV and pyqt5 online documentation
    def display_image(self, imageType, window=1):
        qformat = QImage.Format_Indexed8  # image is stored using 8-bit indexes into a colourmap


        if imageType is "source":
            if len(self.sourceImage.shape) == 3:  # rows[0] cols[1] #channels[2]
                if (self.sourceImage.shape[2]) == 4:  # if there are 4 channels
                    qformat = QImage.Format_RGBA8888  # image stored using 32-bit format
                else:
                    qformat = QImage.Format_RGB888  # image stored using 24-bit format

            # Assign img to QImage object of the loaded image with correct format
            img = QImage(self.sourceImage, self.sourceImage.shape[1], self.sourceImage.shape[0],
                         self.sourceImage.strides[0], qformat)
            img = img.rgbSwapped()  # Transform image from RGB to BGR
            if window is 1:
                self.sourceImgLabel.setPixmap(QPixmap.fromImage(img))
                self.sourceImgLabel.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)  # Centre image in label

        elif imageType is "destination":
            if len(self.dstImage.shape) == 3:  # rows[0] cols[1] #channels[2]
                if (self.dstImage.shape[2]) == 4:  # if there are 4 channels
                    qformat = QImage.Format_RGBA8888  # image stored using 32-bit format
                else:
                    qformat = QImage.Format_RGB888  # image stored using 24-bit format

            # Assign img to QImage object of the loaded image with correct format
            img = QImage(self.dstImage, self.dstImage.shape[1], self.dstImage.shape[0], self.dstImage.strides[0],
                         qformat)
            img = img.rgbSwapped()  #Transform image from RGB to BGR
            if window is 1:
                self.sourceImgLabel.setPixmap(QPixmap.fromImage(img))
                self.sourceImgLabel.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)  # Centre image in label


        elif imageType is "output":
            if len(self.outputImage.shape) == 3:  # rows[0] cols[1] #channels[2]
                if (self.outputImage.shape[2]) == 4:  # if there are 4 channels
                    qformat = QImage.Format_RGBA8888  # image stored using 32-bit format
                else:
                    qformat = QImage.Format_RGB888  # image stored using 24-bit format

            # Assign img to QImage object of the loaded image with correct format
            img = QImage(self.outputImage, self.outputImage.shape[1], self.outputImage.shape[0],
                         self.outputImage.strides[0], qformat)
            img = img.rgbSwapped()  # Transform image from RGB to BGR
            if window is 1:
                self.sourceImgLabel.setPixmap(QPixmap.fromImage(img))
                self.sourceImgLabel.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)  # Centre image in label

    ##display_image##


    # movement coordinates
    # When their corresponding button in the GUI is selected
    # they can be defined by clicking on the image
    @pyqtSlot()
    def select_mc1(self):
        self.ground.plane.movementCoordID = 0
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False
        self.ground.plane.lowestPointOn = False

    @pyqtSlot()
    def select_mc2(self):
        self.ground.plane.movementCoordID = 1
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False
        self.ground.plane.lowestPointOn = False

    @pyqtSlot()
    def select_mc3(self):
        self.ground.plane.movementCoordID = 2
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False
        self.ground.plane.lowestPointOn = False

    @pyqtSlot()
    def select_mc4(self):
        self.ground.plane.movementCoordID = 3
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False
        self.ground.plane.lowestPointOn = False

    @pyqtSlot()
    def select_mc5(self):
        self.ground.plane.movementCoordID = 4
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False
        self.ground.plane.lowestPointOn = False

    @pyqtSlot()
    def finish_mc_selection(self):
        self.ground.plane.movementCoordID = None
        self.ground.plane.movementDefined = True

    # Plane lines in image
    # When their corresponding button in the GUI is selected
    # they can be defined by clicking on the image
    @pyqtSlot()
    def select_line1(self):
        self.ground.plane.lineID = 0
        self.ground.plane.lowestPointOn = False
        self.ground.plane.midpointOn = False

    @pyqtSlot()
    def select_line2(self):
        self.ground.plane.lineID = 1
        self.ground.plane.lowestPointOn = False
        self.ground.plane.midpointOn = False

    @pyqtSlot()
    def select_line3(self):
        self.ground.plane.lineID = 2
        self.ground.plane.lowestPointOn = False
        self.ground.plane.midpointOn = False

    @pyqtSlot()
    def select_midpoint(self):
        self.ground.plane.midpointOn = True
        self.ground.plane.lowestPointOn = False
        self.ground.plane.lineID = None

    @pyqtSlot()
    def finish_line_selection(self):
        self.ground.plane.isDefined = True
        self.ground.plane.lineID = None
        self.ground.plane.midpointOn = False

    @pyqtSlot()
    def select_lowest_point(self):
        self.ground.plane.lowestPointOn = True
        self.ground.plane.midpointOn = False
        self.ground.plane.lineID = None



    # Handles mouse clicks in the image
    def mousePressEvent(self, event):

        # If a line button has been selected
        if self.ground.plane.lineID is not None:  # if line has been selected
            qpoint = self.sourceImgLabel.mapFrom(self, event.pos())  # Locate click co-ordinates on imgLabel

            if (qpoint.x() >= 0 and qpoint.y() >= 0 and qpoint.x() <= self.sourceImage.shape[1]
                    and qpoint.y() <= self.sourceImage.shape[0]):  # Boundary checking

                point = tuple((qpoint.x(), qpoint.y()))

                # if self.plane.lines[self.plane.lineID].startPoint is None:
                if self.ground.plane.lines[self.ground.plane.lineID].pointNum is 1:

                    # Populate line
                    self.ground.plane.lines[self.ground.plane.lineID].startPointUV = point
                    self.ground.plane.lines[self.ground.plane.lineID].pointNum = 2
                    self.ground.plane.lines[self.ground.plane.lineID].isDefined = False
                    self.ground.plane.lines[self.ground.plane.lineID].id = self.ground.plane.lineID

                # elif self.plane.lines[self.plane.lineCount].endPoint is None:
                elif self.ground.plane.lines[self.ground.plane.lineID].pointNum is 2:

                    # Checking if point 2 is the same as point 1
                    if point != self.ground.plane.lines[self.ground.plane.lineID].startPointUV:
                        # Populate line
                        self.ground.plane.lines[self.ground.plane.lineID].endPointUV = point
                        self.ground.plane.lines[self.ground.plane.lineID].isDefined = True
                        self.ground.plane.lines[self.ground.plane.lineID].e2h()
                        self.ground.plane.lines[self.ground.plane.lineID].line_equation()
                        self.draw_line(self.ground.plane.lines[self.ground.plane.lineID])
                        self.ground.plane.lines[self.ground.plane.lineID].pointNum = 1

        # If lowest point button has been selected
        elif self.ground.plane.lowestPointOn is not False:
            qpoint = self.sourceImgLabel.mapFrom(self, event.pos())  # Locate click co-ordinates on imgLabel

            if (qpoint.x() >= 0 and qpoint.y() >= 0 and qpoint.x() <= self.sourceImage.shape[1]
                    and qpoint.y() <= self.sourceImage.shape[0]):  # Boundary checking

                point = [qpoint.x(), qpoint.y()]

                # Set lowest point
                self.ground.plane.lowestPoint = point
                cv2.circle(self.sourceImage, (int(point[0]), int(point[1])), 3, (0, 0, 255), 3)
                self.display_image("source", 1)
                print(self.ground.plane.lowestPoint)

        # If midpoint button has been selected
        elif self.ground.plane.midpointOn is True:
            qpoint = self.sourceImgLabel.mapFrom(self, event.pos())  # Locate click co-ordinates on imgLabel

            if (qpoint.x() >= 0 and qpoint.y() >= 0 and qpoint.x() <= self.sourceImage.shape[1]
                    and qpoint.y() <= self.sourceImage.shape[0]):  # Boundary checking

                # Set midpoint
                point = tuple((qpoint.x(), qpoint.y()))
                self.ground.plane.midpoint = point
                cv2.circle(self.sourceImage, (int(point[0]), int(point[1])), 3, (255, 0, 0), 3)
                self.display_image("source", 1)

        # if movement coordinate button has been selected
        elif self.ground.plane.movementCoordID is not None:
            qpoint = self.sourceImgLabel.mapFrom(self, event.pos())  # Locate click co-ordinates on imgLabel

            if (qpoint.x() >= 0 and qpoint.y() >= 0 and qpoint.x() <= self.sourceImage.shape[1]
                    and qpoint.y() <= self.sourceImage.shape[0]):  # Boundary checking

                # Set movement coordinate
                point = [qpoint.x(), qpoint.y()]
                cv2.circle(self.sourceImage, (int(point[0]), int(point[1])), 3, (0, 0, 255), 3)
                cv2.putText(self.sourceImage, str((self.ground.plane.movementCoordID + 1)),
                            (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                            cv2.LINE_AA)
                self.display_image("source", 1)
                self.ground.plane.movementCoords[self.ground.plane.movementCoordID] = qpoint

                print(self.ground.plane.movementCoords[self.ground.plane.movementCoordID])

    ##mousePressEvent##


    def draw_line(self, line):
        cv2.line(self.sourceImage, line.startPointUV, line.endPointUV, (0, 255, 0), 2)
        self.display_image("source", 1)
        print(line.startPointH, line.endPointH)

    ##draw_line##


    # This functions calls the functions that calculate the corresponding
    # points in the image and the model
    def calc_correspondingPts(self):

        # Reseting values
        corresCoordID = 0
        self.ground.plane.corPts_src.clear()
        self.ground.plane.corPts_dst.clear()
        self.ground.plane.intersectionCoordsE.clear()
        self.ground.plane.intersectionCoordsH.clear()

        # If the plane has been defined
        if self.ground.plane.isDefined is True:
            self.ground.plane.line_intersections()  # Calculate the intersection of user defined lines

            # Draw a circle at each intersection
            for point in self.ground.plane.corPts_src:
                cv2.circle(self.sourceImage, point, 5, (255, 0, 0), 3)
                corresCoordID += 1

            # Display image with points
            self.display_image("source", 1)

            # Reset lines to undefined
            for line in self.ground.plane.lines:
                line.isDefined = False

            # Calculate corresponding points in model
            self.ground.plane.calculate_corresPts(width=self.ground.width, length=self.ground.length)

    ##calc_intersections


    # Find homography using corresponding points
    def find_homography(self):

        # Source (image) points being formatted into numpy array
        pts_src_na = np.array(self.ground.plane.corPts_src)
        # Destination (model) points being formatted into numpy array
        pts_dst_na = np.array(self.ground.plane.corPts_dst)

        # Calculate Homography and texture extraction homography
        self.ground.plane.H, status = cv2.findHomography(pts_src_na, pts_dst_na)
        self.ground.plane.texH, status = cv2.findHomography(pts_src_na, pts_dst_na)

        # Clear corresponding points and undefine plane
        self.ground.plane.corPts_src.clear()
        self.ground.plane.corPts_dst.clear()
        self.ground.plane.isDefined = False

    ##find_homography##


    # Tranforms points by homography matrix
    # Parameter: 'x' the x coordinate to be transformed
    # Parameter: 'y' the y coordinate to be transformed
    # Parameter: 'H' the homography matrix
    # Parameter: 'reverse' whether to invert H or not
    def transformPoints(self, x, y, H, reverse=False):
        # Invert if true
        if reverse is True:
            val, H = cv2.invert(H)

        # get the elements from H for transformation equation
        h0 = H[0, 0]
        h1 = H[0, 1]
        h2 = H[0, 2]
        h3 = H[1, 0]
        h4 = H[1, 1]
        h5 = H[1, 2]
        h6 = H[2, 0]
        h7 = H[2, 1]
        h8 = H[2, 2]

        # Transformation equation
        tx = (h0 * x + h1 * y + h2)
        ty = (h3 * x + h4 * y + h5)
        tz = (h6 * x + h7 * y + h8)

        # New projected coordinates
        px = tx / tz
        py = ty / tz
        Z = int(1 / tz)

        return [px, py]

    ##transform_points##


    #Performs all the steps to make a meshgrid
    @pyqtSlot()
    def implement_meshgrid(self):

        # Checking for prerequisites
        if self.ground.plane.isDefined is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Lines have not been selected')
        elif self.ground.plane.lowestPoint is None:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Lowest Point of plane has not been selected')
        elif self.ground.dimDefined is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Room dimensions have not been specified')
        elif self.ground.plane.visLengthPercentage is None:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Percentage must be given')
        elif self.ground.plane.midpoint is None:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Midpoint has not been selected')

        else:
            self.errorOutTextEdit.clear()

            # Calculating points for homography
            # Finding homography
            # Creating and mapping meshgrid
            self.calc_correspondingPts()
            self.find_homography()
            self.ground.meshgrid = self.createMeshgrid()
            self.ground.map_meshgrid(self.ground.plane.H)

            self.meshgridCreated = True
            self.textureExtracted = False

            self.currentOperationTextEdit.clear()
            self.currentOperationTextEdit.append('1.) Draw Meshgrid')

    ##implement_meshgrid##


    # Draws meshgrid on the image
    @pyqtSlot()
    def draw_meshgrid(self):

        if self.meshgridCreated is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Meshgrid has not been created')
        else:
            self.ground.draw_meshgrid(self.gridImage)
            self.sourceImage = self.gridImage
            self.display_image("source", 1)
            self.currentOperationTextEdit.clear()
            self.currentOperationTextEdit.append('1.) Extract Texture')
    ##draw_meshgrid##


    # Mouse Callback for ROI selection
    # event: The event that took place (left or right mouse click)
    # Parameter: 'x' the x coordinate of the event.
    # Parameter: 'y' the y coordinate of the event.
    # flags: Any relevant flags passed by OpenCV.
    # params: Any extra parameters supplied by OpenCV.
    def roi_selection(self, event, x, y, flags, param):

        # On Left mouse button click records roi with mouse selection status to True
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_selection = True
            self.texture_roi = [x, y, x, y]
        # On Mouse movement records roi with mouse selection status to True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_selection == True:
                self.texture_roi[2] = x
                self.texture_roi[3] = y

        # If Left mouse button is released changes mouse selection status to False
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_selection = False
            self.texture_roi[2] = x
            self.texture_roi[3] = y

    ##roi_selection##


    # Extract texture from image
    def extract_texture(self):

        # If the furthest left point of the ground in the image is within a certain range
        # shift homography tanslation
        if ((self.sourceImage.shape[1] / self.ground.plane.leftMostPoint[0]) < 4):
            self.ground.plane.texH[0, 2] = self.ground.plane.texH[0, 2] + 100

        # If the furthest left point of the ground in the image is within a certain range
        # shift homography tanslation
        if ((self.sourceImage.shape[0] / self.ground.plane.topMostPoint[1]) > 3):
            self.ground.plane.texH[1, 2] = self.ground.plane.texH[1, 2] + 200

        # Warp source image to destination based on homography
        self.outputImage = cv2.warpPerspective(self.textureImage, self.ground.plane.texH,
                                               (self.textureImage.shape[1] * 2, self.textureImage.shape[0] * 2))

        # Original Image Window Name
        window_name = 'Texture Extractor'

        # Cropped Image Window Name
        window_crop_name = 'Cropped Texture'

        # Escape ASCII Keycode
        esc_keycode = 13

        # Time to waitfor
        wait_time = 1

        # Check if image is loaded
        if self.outputImage is not None:
            # Make a copy of original image for cropping
            clone = self.outputImage.copy()
            # Create a Window
            # cv2.WINDOW_NORMAL = Enables window to resize.
            # cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # Set mouse handler for Window with roi_selection function callback
            cv2.setMouseCallback(window_name, self.roi_selection)

            while True:
                # Show original image in window
                cv2.imshow(window_name, self.outputImage)

                # if roi has all parameters filled
                if len(self.texture_roi) == 4:
                    # Make a copy of orginal image before drawing rectangle on it
                    self.outputImage = clone.copy()
                    # Check if any pixel coordinate is negative and make it zero
                    self.texture_roi = [0 if i < 0 else i for i in self.texture_roi]

                    # Draw rectangle on input_img
                    cv2.rectangle(self.outputImage, (self.texture_roi[0], self.texture_roi[1]), (self.texture_roi[2],
                                                                                                 self.texture_roi[3]),
                                  (0, 255, 0), 2)
                    # Make x and y coordiates for cropping in ascending order
                    # if x1 = 200,x2= 10 make x1=10,x2=200
                    if self.texture_roi[0] > self.texture_roi[2]:
                        x1 = self.texture_roi[2]
                        x2 = self.texture_roi[0]
                    # else keep it as it is
                    else:
                        x1 = self.texture_roi[0]
                        x2 = self.texture_roi[2]
                    # if y1 = 200,y2= 10 make y1=10,y2=200
                    if self.texture_roi[1] > self.texture_roi[3]:
                        y1 = self.texture_roi[3]
                        y2 = self.texture_roi[1]
                    # else keep it as it is
                    else:
                        y1 = self.texture_roi[1]
                        y2 = self.texture_roi[3]

                    # Crop clone image
                    crop_img = clone[y1: y2, x1: x2]
                    # check if crop_img is not empty
                    if len(crop_img):
                        # Create a cropped image Window
                        cv2.namedWindow(window_crop_name, cv2.WINDOW_AUTOSIZE)
                        # Show image in window
                        if (crop_img.shape[0] > 5 and crop_img.shape[1] > 5):
                            cv2.imshow(window_crop_name, crop_img)

                # Check if any key is pressed
                k = cv2.waitKey(wait_time)
                # Check if ESC key is pressed. ASCII Keycode of ESC=27
                if k == esc_keycode:
                    cv2.imwrite(
                        r'D:\PyCharm\PycharmProjects\ProjectGUI\venv\Lib\site-packages\vpython\vpython_data\ground_texture.jpg',
                        crop_img)
                    # cv2.waitKey(0)
                    self.ground.modelLength = crop_img.shape[0]
                    print(self.ground.modelLength)
                    # Destroy All Windows
                    cv2.destroyAllWindows()
                    break
    ##extract_texture##


    # Calls the extract_texture function
    @pyqtSlot()
    def texture_extraction(self):

        # Checking for meshgrid
        if self.meshgridCreated is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Meshgrid has not been created')
        else:

            # Extract texture
            self.extract_texture()
            self.textureExtracted = True
            self.currentOperationTextEdit.clear()
            self.currentOperationTextEdit.append('1.) Create 3D Model')
    ##texture_extraction##


    # Create 3D model
    @pyqtSlot()
    def create_model_3D(self):

        # Checking for texture
        if self.textureExtracted is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Texture has not been extracted')
        else:
            # If model is being recreated recreate canvas
            if self.modelCreated is True:
                self.scene.delete()
                self.scene = canvas(width=1000, height=1000)
            else:
                # Create canvas
                self.scene = canvas(width=1000, height=1000)

            # if ground dimensions have been defined
            if self.ground.dimDefined is True:

                # Ground dimensions
                groundY = self.ground.length
                groundX = self.ground.width

                # Thickness of axis arrows and grid lines depending on dimensions
                if groundX < 200 or groundY < 200:
                    arrowWidth = 0.5
                    lineThickness = 1
                else:
                    arrowWidth = 1
                    lineThickness = 2

                # Offset scene centre
                self.scene.centre = vector(-groundX / 2, -groundY / 2, 0)

                # Make scene invisible until model has loaded
                self.scene.visible = False

                # Lighting
                self.scene.lights = []
                distant_light(direction=vector(0, -1, 1), color=color.white)

                # Calculate the model coordinate for the lowest point
                lowsetCorner = self.transformPoints(self.ground.plane.lowestPoint[0],
                                                    self.ground.plane.lowestPoint[1], self.ground.plane.texH)

                # Set length of model to lowest coordinate Y value
                self.ground.modelLength = self.ground.length - int(lowsetCorner[1])

                # Create model
                self.ground.model = box(pos=self.scene.centre + vector(groundX / 2, (self.ground.modelLength / 2)
                                                                       + (groundY - self.ground.modelLength), -3),
                                        size=vector(groundX, self.ground.modelLength, 3),
                                        texture={'file': 'ground_texture.jpg', 'flipy': True}, opacity=1)

                # Create horizontal meshgrid lines for model
                for line in self.ground.meshgrid.lines_horizontal:
                    line3D = shapes.line(
                        start=(int(line.startPoint_cart[0] - groundX / 2), int(line.startPoint_cart[1] - groundY / 2)),
                        end=(int(line.endPoint_cart[0] - groundX / 2),
                             int(line.endPoint_cart[1] - groundY / 2)), thickness=lineThickness, np=10)

                    self.ground.meshgrid.lines_horizontal3D.append(line3D)

                # Create vertical meshgrid line for model
                for line in self.ground.meshgrid.lines_vertical:
                    line3D = shapes.line(
                        start=(int(line.startPoint_cart[0] - groundX / 2), int(line.startPoint_cart[1] - groundY / 2)),
                        end=(int(line.endPoint_cart[0] - groundX / 2),
                             int(line.endPoint_cart[1] - groundY / 2)), thickness=lineThickness, np=10)

                    self.ground.meshgrid.lines_vertical3D.append(line3D)

                # Draw the horizontal lines
                for line in self.ground.meshgrid.lines_horizontal3D:
                    extrusion(path=[vec(0, 0, 0), vec(0, 0, 0.1)],
                              shape=line)

                # Draw the vertical lines
                for line in self.ground.meshgrid.lines_vertical3D:
                    extrusion(path=[vec(0, 0, 0), vec(0, 0, 0.1)],
                              shape=line)

                # Create axis arrows
                xAxis = arrow(pos=self.scene.centre, axis=vector(groundX + groundX / 3, 0, 0), shaftwidth=arrowWidth)
                yAxis = arrow(pos=self.scene.centre, axis=vector(0, groundY + groundY / 3, 0), shaftwidth=arrowWidth)
                zAxis = arrow(pos=self.scene.centre, axis=vector(0, 0, -groundY), shaftwidth=arrowWidth)

                # Onscreen text
                text1 = text(text='Controls', align='centre', color=color.blue, height=groundY / 13,
                             billboard=True, depth=0, pos=vector(groundX / 2 + groundX / 4, groundY + groundY / 4, 0),
                             emissive=True)

                text2 = text(text="'q' to quit loop", align='left', color=color.blue, height=groundY / 20,
                             billboard=True, depth=0, pos=vector(text1.pos - vector(0, groundY / 12, 0)),
                             emissive=True)

                text1 = text(text="'d' to delete scene", align='left', color=color.blue, height=groundY / 20,
                             billboard=True, depth=0, pos=vector(text2.pos - vector(0, groundY / 12, 0)),
                             emissive=True)

                self.modelCreated = True

                # wait for texture to load then make model visible
                self.scene.waitfor("textures")
                self.scene.visible = True

                running = True

                while running:

                    ev = self.scene.waitfor('mousedown keydown')

                    if ev.event == 'mousedown':
                        print(ev.pos)  # the position of the mouse

                        # Projecting mouse onto the x-y plane of the model
                        m_posXY = self.scene.mouse.project(normal=vector(0, 0, 1))

                        # Reset top front facing view
                        self.scene.forward = vector(0, 0, -1)

                    if ev.event == 'keydown':

                        # Quit loop if q is pressed
                        if ev.key == 'q':
                            print("q")
                            running = False

                            self.currentOperationTextEdit.clear()
                            self.currentOperationTextEdit.append('1.) Select Location Coordinates\n2.) Map Locations')

                        # Delete scene is d is pressed
                        elif ev.key == 'd':
                            print("d")
                            self.scene.delete()
                            self.modelCreated = False
                            self.scene = None
                            running = False
                            self.currentOperationTextEdit.clear()
                            self.currentOperationTextEdit.append(
                                '1.) Create 3D model\nNote: Extract texture again if required')

    ##create_model_3D##


    #Calls map movement
    @pyqtSlot()
    def movement_map(self):

        if self.modelCreated is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('3D model has not been created')

        else:
            #Display image with meshgrid
            self.sourceImage = self.gridImage
            self.display_image("source", 1)
            self.map_movement()

    ##load_clicked##


    def map_movement(self):

        if self.ground.plane.movementDefined is False:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('Movement coordinates have not been defined')

        elif self.scene is None:
            self.errorOutTextEdit.clear()
            self.errorOutTextEdit.append('3D model must be created')
        else:

            self.errorOutTextEdit.clear()
            self.currentOperationTextEdit.clear()

            modelLocations = []  #list of locations on model

            #Tranform the coordinates in the image to the model coordinates
            #and store in modelLocations
            for point in self.ground.plane.movementCoords:

                #If the point has been selected
                if point.isNull() is False:
                    location = self.transformPoints(point.x(), point.y(), self.ground.plane.H)
                    modelLocations.append(location)


            #If there are locations to be mapped
            if len(modelLocations) > 0:

                #Position of first location
                posX = int(modelLocations[0][0]) - self.ground.width/2
                posY = int(modelLocations[0][1]) - self.ground.length/2


                #Create a box at the first location
                box1 = box(pos=vector(posX, posY, 0), size=vector(self.ground.width/self.ground.meshgrid.xDimension/2,
                     self.ground.length / self.ground.meshgrid.yDimension/2,(self.ground.length / self.ground.meshgrid.yDimension)/2)
                       ,color = color.red, make_trail = True)

                #Trail color that the box leaves behind
                box1.trail_color = color.green

                #On screen text
                animationText = text(text="Press 'r' to run animation", align='left', color=color.green, height=self.ground.length / 10,
                         billboard=True, depth=0, pos=vector(-self.ground.width, self.ground.length + self.ground.length / 3, 0),
                         emissive=True)


                running = True

                #Animation loop
                while running:

                    #wait for user input
                    ev = self.scene.waitfor('mousedown keydown')

                    #If a key is pressed
                    if ev.event == 'keydown':

                        #quit loop if q is pressed
                        if ev.key == 'q':
                            print("q")
                            self.currentOperationTextEdit.clear()
                            running = False

                        #delete the scene if d is pressed
                        elif ev.key == 'd':
                            print("d")
                            self.currentOperationTextEdit.clear()
                            self.scene.delete()
                            self.modelCreated = False
                            running = False

                        #run the animation if r is pressed
                        elif ev.key == 'r' and len(modelLocations) > 0:

                            #from the 2nd to the last model location
                            for i in range(1, len(modelLocations)):

                                #Vector of the target location
                                locationVector = vector(modelLocations[i][0] - (self.ground.width / 2),
                                                    modelLocations[i][1] -
                                                    (self.ground.length / 2), 0)

                                #vector from the box to the target location
                                moveVector = locationVector - box1.pos

                                #normalize the moveVector
                                moveVectorNorm = moveVector.norm()

                                #Create a small blue at the target location
                                box2 = box(pos=vector(locationVector.x, locationVector.y, 0), size= box1.size/3,
                                      color=color.blue)

                                #distance from the box to target location
                                distance = mag(moveVector)

                                #while the box hasnt reached the target location
                                while distance > 1:

                                    rate(50)

                                    #Move the box
                                    box1.pos += (moveVectorNorm)

                                    #Recalculate moveVector
                                    moveVector = locationVector - box1.pos

                                    # Recalculate the distance
                                    distance = mag(moveVector)
                                    print(distance)

                                #Delete the blue box when the original box reaches it
                                box2.visible = False
                                del box2

                            #clear old locations
                            modelLocations.clear()

                #Undefine image locations
                self.ground.plane.movementDefined = False

                #Reset the image locations
                for coord in self.ground.plane.movementCoords:
                    coord.setX(0)
                    coord.setY(0)

                #Delete the movement box
                box1.visible = False
                box1.clear_trail()
                del box1
            else:
                self.errorOutTextEdit.clear()
                self.errorOutTextEdit.append('Movement coordinates have not been defined')

    ##map_movement##


    #Create meshgrid in model space
    def createMeshgrid(self):

        #Create meshgrid with 10 x values and 10 y values
        #i_coords = matrix that contains all possible x values for all combinations with y values
        #j_coords = matrix that contains all possible y values for all combinations with x values
        i_coords, j_coords = np.meshgrid(range(11), range(11), indexing = 'ij')

        #Create coordinate array by putting both matrices in an array
        coordinate_grid = np.array([i_coords, j_coords])

        #The x and y incremental value for the coorinates in the grid
        xIncrement = int(round(self.ground.width / 10, 0));
        yIncrement = int(round(self.ground.length / 10, 0));

        #Setting all values to 0
        coordinate_grid[:,:,:] = 0


        #Setting all the values for the coordiantes needed to define the start and
        #endpoints for the lines in the meshgrid. (All values at the border of the grid)


        #list of Y values on left of grid at x = 0
        lYvalues = []
        scale = 0


        #Creating Y values on left of grid at x = 0
        for value in coordinate_grid[1,0, :]:
            value += yIncrement * scale #Increment value
            lYvalues.append(value) #Add value to list
            scale += 1

        #Setting coordinate_grid to correct values
        coordinate_grid[1,0, :]  = lYvalues


        # list of Y values on right of grid at x = 9
        rYvalues = []
        scale = 0

        # Creating Y values on right
        for value in coordinate_grid[1, 10, :]:
            value += yIncrement * scale #Increment value
            rYvalues.append(value) #Add value to list
            scale += 1

        # Setting coordinate_grid to correct values
        coordinate_grid[1, 10, :] = rYvalues


        # list of Y values on top of grid from x = 0 to x = 9
        topYvalues = []
        scale = 0

        # Creating Y values on top of grid at y = 9 for x = 0 to x = 9
        for value in coordinate_grid[1, :, 10]:
            value = self.ground.length
            topYvalues.append(value) #Add value to list

        #Setting coordinate_grid to correct values
        coordinate_grid[1, :, 10] = topYvalues


        #list of X values on bottom of grid at y = 0
        botXvalues = []
        scale = 0

        # Creating all X values on bottom at y = 9
        for value in coordinate_grid[0, :, 0]:
            value += xIncrement * scale
            botXvalues.append(value)
            scale += 1

        # Setting coordinate_grid to correct values
        coordinate_grid[0, :, 0] = botXvalues


        # list of X values on bottom of grid at y = 9
        topXvalues = []
        scale = 0

        # Creating all X values on top at y = 9
        for value in coordinate_grid[0, :, 10]:
            value += xIncrement * scale
            topXvalues.append(value)
            scale += 1

        # Setting coordinate_grid to correct values
        coordinate_grid[0, :, 10] = topXvalues


        # list of X values on bottom of grid at y = 10
        rXvalues = []
        scale = 0

        # Creating X values on right at x = 10
        for value in coordinate_grid[0, 10, :]:
            value = self.ground.width
            rXvalues.append(value)

        # Setting coordinate_grid to correct values
        coordinate_grid[0, 10, :] = rXvalues


        # lists of vertical and horiziontal line
        # start and endpoints that will make the grid
        v_startPoint = coordinate_grid[:, :, 10]
        v_endPoint = coordinate_grid[:, :, 0]
        h_startPoint = coordinate_grid[:, 0, :]
        h_endPoint = coordinate_grid[:, 10, :]

        #Create meshgrid
        meshgrid = Meshgrid()

        #Populate meshgrids horizontal lines
        for index in range(11):
            startPoint = [h_startPoint[0][index], h_startPoint[1][index]]
            endPoint = [h_endPoint[0][index], h_endPoint[1][index]]

            line = Line(startPoint_cart= startPoint, endPoint_cart= endPoint)
            meshgrid.lines_horizontal.append(line)

        #Populate meshgrids vertical lines
        for index in range(11):
            startPoint = [v_startPoint[0][index], v_startPoint[1][index]]
            endPoint = [v_endPoint[0][index], v_endPoint[1][index]]

            line = Line(startPoint_cart=startPoint, endPoint_cart=endPoint)
            meshgrid.lines_vertical.append(line)


        return meshgrid

    ##Create_Meshgrid##

##GUI##

#Meshgrid contains all grid lines for the ground
#in the image and the model
class Meshgrid:
    def __init__(self):
        self.lines_horizontal = [] #Model space lines
        self.lines_vertical = []   #Model space lines
        self.xDimension = 10;
        self.yDimension = 10;
        self.lines_horizontal3D = []  # 3D Model lines to be drawn
        self.lines_vertical3D = []    #3D Model lines to be drawn
        self.lines_horizontal_img = [] #Image lines
        self.lines_vertical_img = []   #Image lines

    ##Meshgrid##


class Ground:
    def __init__(self, **kwargs):
        self.plane = Plane()   #ground plane
        self.model = None      #ground model
        self.meshgrid = None   #ground meshgrid
        self.dimDefined = False  # are the dimensions defined
        self.width = 0
        self.length = 0
        self.modelLength = 0
        self.isLonger = False
        self.isWider = False

    ##Ground##


    #Map the meshgrid from the model to the image
    #Parameter: 'H' the homography matrix
    def map_meshgrid(self, H):

        #invert homography to back-project points
        #from model to image
        val, Hinv = cv2.invert(H)

        #get elements from H for the transformation equation
        h0 = Hinv[0, 0]
        h1 = Hinv[0, 1]
        h2 = Hinv[0, 2]
        h3 = Hinv[1, 0]
        h4 = Hinv[1, 1]
        h5 = Hinv[1, 2]
        h6 = Hinv[2, 0]
        h7 = Hinv[2, 1]
        h8 = Hinv[2, 2]


        #Transform all the start and end points for the horizontal lines
        for line in self.meshgrid.lines_horizontal:

            startPointX = line.startPoint_cart[0]
            startPointY = line.startPoint_cart[1]
            endPointX = line.endPoint_cart[0]
            endPointY = line.endPoint_cart[1]

            #Transform for startPoint coords
            txS = (h0 * startPointX + h1 * startPointY + h2)
            tyS = (h3 * startPointX + h4 * startPointY + h5)
            tzS = (h6 * startPointX + h7 * startPointY + h8)

            #Transform for endPoint coords
            txE = (h0 * endPointX + h1 * endPointY + h2)
            tyE = (h3 * endPointX + h4 * endPointY + h5)
            tzE = (h6 * endPointX + h7 * endPointY + h8)

            #Projected startPoint coords
            pxS = txS / tzS
            pyS = tyS / tzS
            ZS = int(1 / tzS)

            #Projected endPoint coords
            pxE = txE / tzE
            pyE = tyS / tzE
            ZE = int(1 / tzE)

            #Projected start and end points
            startPointP = [pxS, pyS, ZS]
            endPointP = [pxE, pyE, ZE]

            #Create new line
            newLine = Line(startPointUV=startPointP, endPointUV=endPointP)

            #Append to meshgrids horizontal image lines
            self.meshgrid.lines_horizontal_img.append(newLine)

        #Transform all the start and end points for the vertical lines
        for line in self.meshgrid.lines_vertical:
            startPointX = line.startPoint_cart[0]
            startPointY = line.startPoint_cart[1]
            endPointX = line.endPoint_cart[0]
            endPointY = line.endPoint_cart[1]

            # Transform for startPoint coords
            txS = (h0 * startPointX + h1 * startPointY + h2)
            tyS = (h3 * startPointX + h4 * startPointY + h5)
            tzS = (h6 * startPointX + h7 * startPointY + h8)

            # Transform for endPoint coords
            txE = (h0 * endPointX + h1 * endPointY + h2)
            tyE = (h3 * endPointX + h4 * endPointY + h5)
            tzE = (h6 * endPointX + h7 * endPointY + h8)

            # Projected startPoint coords
            pxS = txS / tzS
            pyS = tyS / tzS
            ZS = int(1 / tzS)

            # Projected endPoint coords
            pxE = txE / tzE
            pyE = tyE / tzE
            ZE = int(1 / tzE)

            # Projected start and end points
            startPointP = [pxS, pyS, ZS]
            endPointP = [pxE, pyE, ZE]

            # Create new line
            newLine = Line(startPointUV= startPointP, endPointUV= endPointP)

            # Append to meshgrids horizontal image lines
            self.meshgrid.lines_vertical_img.append(newLine)

    ##map_meshgrid##


    #Draw the meshgrid on the image
    #Parameter: 'image' image to be drwn on
    def draw_meshgrid(self, image):

        for line in self.meshgrid.lines_horizontal_img:
            cv2.line(image, (int(line.startPointUV[0]), int(line.startPointUV[1]))
                     , (int(line.endPointUV[0]), int(line.endPointUV[1]))  , (0, 0, 0), 1)

        for line in self.meshgrid.lines_vertical_img:
            cv2.line(image, (int(line.startPointUV[0]), int(line.startPointUV[1]))
                     , (int(line.endPointUV[0]), int(line.endPointUV[1])), (0, 0, 0), 1)
    ##draw_meshgrid##

##Ground##



class Plane:
    def __init__(self):
        self.lines = [Line(), Line(), Line()]  #User selected lines for ground plane triangle
        self.lineCount = 0
        self.isDefined = False
        self.lineID = None      #Used user is defining lines
        self.midpointOn = False  #Used user is defining midpoint
        self.midpoint = None     #Midpoint of plane on the edge that define quadrilateral
        self.intersectionCoordsH = []  #Homogeneous intersection coordinates of lines
        self.intersectionCoordsE = []  #Euclidean intersection coordinates of lines
        self.corPts_src = []    #Source (image) corresponding points
        self.corPts_dst = []    #Destination (model) corresponding points
        self.lowestPoint = None  #lowest coordinate of the ground in the image
        self.lowestPointOn = False   #Used user is defining lowestPoint
        self.leftMostPoint = None    #Furthest left coordinate on ground in the image
        self.topMostPoint = None     #Highest coordinate on ground in the image
        self.H = None               #Homography matrix between image and model plane
        self.texH = None            #Homography matrix used for texture extraction
        self.movementCoords = [QPoint(), QPoint(),  QPoint(),  QPoint(),  QPoint()]  #Coordinates of locations in the image that
                                                                                     #that are used for movement animation on model
        self.movementCoordID = None     #Used user is defining movement coordinates
        self.movementDefined = False
        self.visLengthPercentage = None  #Visible percentage of the ground in the image on the camera's side


    #Calculates the intersections of the lines in the image
    def line_intersections(self):


        l_intersectionCoordsH = []  # local homogenous coordinate list
        l_intersectionCoordsE = []  # local euclidean coordinate list


        #Find intersection points of lines
        for lineA in self.lines:

            for lineB in self.lines:

                if lineA.isDefined is True and lineB.isDefined is True:

                    if lineA is not lineB:
                        point = lineA.equation.cross(lineB.equation)  # point is cross product of two lines

                        l_intersectionCoordsH.append(point)  # add to local list


        l_intersectionCoordsH = list(OrderedDict.fromkeys(l_intersectionCoordsH))  # Remove duplicates
        l_intersectionCoordsE = self.h2e(l_intersectionCoordsH)  # Populate Euclidean coordinates list
        l_intersectionCoordsE = list(OrderedDict.fromkeys(l_intersectionCoordsE))  # Remove duplicates

        self.sort_intersectionCoords(l_intersectionCoordsE)   #Sort coordinates into correct order
        self.intersectionCoordsH = self.e2h(self.intersectionCoordsE) #Populate planes homogenous list


        leftMostX = self.intersectionCoordsE[0][0]
        topMostY = self.intersectionCoordsE[0][1]
        self.topMostPoint = self.intersectionCoordsE[0]
        self.leftMostPoint = self.intersectionCoordsE[0]

        #Calculate the furtherst left and hight point
        for coord in self.intersectionCoordsE:

            if coord[0] < leftMostX:
                self.leftMostPoint = coord
                leftMostX = coord[0]
            if coord[1] < topMostY:
                self.topMostPoint = coord
                topMostY = coord[1]

        #Populate cource corresponding points
        self.corPts_src = self.intersectionCoordsE
        self.corPts_src.append(self.midpoint) #Add midpoint

        i = 0

        # cast intersection coords to integers
        for coord in self.corPts_src:
            self.corPts_src[i] = (int(coord[0]), int(coord[1]))
            i += 1

    ##line_intersections##


    #Sort intersection coordinates into correct order
    def sort_intersectionCoords(self, _intersectionCoords):


        #Sorts the intersection coordinates by finding the lowest coordinate
        #then calculting the closest and furthest coordinate from the lowest
        #Order is then always (closest, lowest, furtherst)
        #Destination corresponding points have the same order

        lowestY = _intersectionCoords[0][1]
        lowestCoord = None
        lowestCoordID = 0

        i = 0;

        for coord in _intersectionCoords:
            if coord[1] > lowestY:
                lowestY = coord[1]
                lowestCoord = coord
                lowestCoordID = i

            i+=1

        distance = 0
        coord1 = None
        coord2 = None

        for id in range(0, 3):

            if id != lowestCoordID:
                if coord1 is None:
                    coord1 = _intersectionCoords[id]
                elif coord2 is None:
                    coord2 = _intersectionCoords[id]


        closestCoord = None
        furthestCoord = None

        distance1 = lowestCoord[0] - coord1[0]
        distance2 = lowestCoord[0] - coord2[0]

        if distance1 < 0:
            distance1 *= -1

        if distance2 < 0:
            distance2 *= -1

        if distance1 < distance2:
            closestCoord = coord1
            furthestCoord = coord2
        elif distance2 < distance1:
            closestCoord = coord2
            furthestCoord = coord1


        self.intersectionCoordsE.append(closestCoord)
        self.intersectionCoordsE.append(lowestCoord)
        self.intersectionCoordsE.append(furthestCoord)

    ##sort_intersectionCoords


    #Calculate the corresponding points on the model
    #Parameter: 'width' the width of the room
    #Parameter: 'length' the length of the room
    def calculate_corresPts(self, width=0, length=0):

        # if Camera is on the left side of the room
        if self.longest_side().m < 0:


            bottomLeftY = length - (length * (self.visLengthPercentage / 100))
            bottomLeft = (0, bottomLeftY)   #length/6.5)
            topLeft = (0, length)
            topRight = (width, length)


            # l_lines = [Line(startPoint_cart=bottomLeft, endPoint_cart=topLeft), Line(startPoint_cart=topLeft,
            #             endPoint_cart=topRight), Line(startPoint_cart=topRight, endPoint_cart=bottomLeft)]

            self.corPts_dst = [topLeft, bottomLeft, topRight, (width,length/2)]



        #if Camera is on the right side of the room
        else:

            bottomRightY = length - (length * (self.visLengthPercentage / 100))

            bottomRight = (width, bottomRightY)
            topRight = (width, length)
            topLeft = (0, length)


            # l_lines = [Line(startPoint_cart=bottomRight, endPoint_cart=topLeft), Line(startPoint_cart=topLeft,
            #              endPoint_cart=topRight),Line(startPoint_cart=topRight, endPoint_cart=bottomRight)]


            self.corPts_dst = [topRight, bottomRight, topLeft, (0, length/2)]

    ##calculate_corresPts##

    #Calculate the longest side of triangle
    def longest_side(self):

        startPoint = None
        endPoint = None

        highestY = self.intersectionCoordsE[0][1]
        startPoint = self.intersectionCoordsE[0]

        # Find startPoint of longest side
        for coord in self.intersectionCoordsE:
            if coord[1] > highestY:
                print("coordY = ", coord[1], "Highest Y = ", highestY)
                startPoint = coord
                highestY = coord[1]

        startVec = Vector(startPoint[0], startPoint[1], 1.0)
        tempVec = None
        endVec = None
        distance = 0

        for coordVec in self.intersectionCoordsH:
            if startVec != coordVec:
                tempVec = startVec.substract(coordVec)
                if (tempVec.magnitude() > distance):
                    endVec = coordVec
                    distance = tempVec.magnitude()

        endPoint = (endVec.x, endVec.y)

        print("StartPoint + endPoint = ", startPoint, endPoint)

        line = Line()

        line.startPointUV = startPoint
        line.endPointUV = endPoint
        line.m = (line.endPointUV[1] - line.startPointUV[1]) / (line.endPointUV[0] - line.startPointUV[0])

        return line

    ##longest_side##


    #Transform the coordinates from homogeneous to euclidean
    #Parameter: 'coordsH' list of homogenous coordinates
    def h2e(self, coordsH):

        l_coordsE = []

        for coord in coordsH:
            x = coord.x / coord.z
            y = coord.y / coord.z
            l_coordsE.append((x, y))

        return l_coordsE

    ##h2e##


    #Transform the coordinates from homogeneous to euclidean
    #Parameter: 'coord' list of euclidean coordinates
    def e2h(self, coords):

        l_coords = []

        for coord in coords:
            l_coords.append(Vector(coord[0], coord[1], 1.0))

        return l_coords

    ##e2h##


##Plane##




class Line:
    def __init__(self, startPointUV=None, endPointUV=None, startPoint_cart=None, endPoint_cart=None):
        self.startPointH = None    #Homogeneous startPoint
        self.endPointH = None       #Homogeneous endPoint
        self.startPointUV = startPointUV      #Screen startPoint
        self.endPointUV = endPointUV          #Screen endPoint
        self.startPoint_cart = startPoint_cart   #Cartesian startPoint
        self.endPoint_cart = endPoint_cart        #Cartesian endPoint
        self.m = None                             #gradient
        self.equation = Vector(0, 0, 0)           #homogeneous equation
        self.isDefined = False
        self.pointNum = 1                         #start and endpoint ID (1 = start, 2 = end)
        self.id = None                            #lineID


    #Transforms lines start and endpoint from euclidean to homogenous
    def e2h(self):
        self.startPointH = Vector(self.startPointUV[0], self.startPointUV[1], 1)
        self.endPointH = Vector(self.endPointUV[0], self.endPointUV[1], 1)
        print(self.startPointH, self.endPointH)

    ##e2h##

    #Calculates homogenous equation of line
    def line_equation(self):
        self.equation = self.startPointH.cross(self.endPointH)

    ##line_equation##


##Line##


#Run GUI
app = QApplication(sys.argv)
window = GUI()
window.setWindowTitle("GUI")
window.show()

sys.exit(app.exec_())
