# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/live_rpi_reconstruction_service.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(504, 534)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_monitor = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_monitor.sizePolicy().hasHeightForWidth())
        self.groupBox_monitor.setSizePolicy(sizePolicy)
        self.groupBox_monitor.setObjectName("groupBox_monitor")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_monitor)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem)
        self.label_3 = QtWidgets.QLabel(self.groupBox_monitor)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.lineEdit_bufferSize = QtWidgets.QLineEdit(self.groupBox_monitor)
        self.lineEdit_bufferSize.setMinimumSize(QtCore.QSize(0, 0))
        self.lineEdit_bufferSize.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_bufferSize.setFont(font)
        self.lineEdit_bufferSize.setReadOnly(True)
        self.lineEdit_bufferSize.setObjectName("lineEdit_bufferSize")
        self.horizontalLayout_6.addWidget(self.lineEdit_bufferSize)
        self.label_4 = QtWidgets.QLabel(self.groupBox_monitor)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_6.addWidget(self.label_4)
        self.lineEdit_processingTime = QtWidgets.QLineEdit(self.groupBox_monitor)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_processingTime.sizePolicy().hasHeightForWidth())
        self.lineEdit_processingTime.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_processingTime.setFont(font)
        self.lineEdit_processingTime.setReadOnly(True)
        self.lineEdit_processingTime.setObjectName("lineEdit_processingTime")
        self.horizontalLayout_6.addWidget(self.lineEdit_processingTime)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButton_on = QtWidgets.QPushButton(self.groupBox_monitor)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_on.setFont(font)
        self.pushButton_on.setCheckable(True)
        self.pushButton_on.setObjectName("pushButton_on")
        self.horizontalLayout_7.addWidget(self.pushButton_on)
        self.pushButton_off = QtWidgets.QPushButton(self.groupBox_monitor)
        self.pushButton_off.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_off.setFont(font)
        self.pushButton_off.setStyleSheet("color: red;")
        self.pushButton_off.setCheckable(True)
        self.pushButton_off.setChecked(True)
        self.pushButton_off.setDefault(False)
        self.pushButton_off.setFlat(False)
        self.pushButton_off.setObjectName("pushButton_off")
        self.horizontalLayout_7.addWidget(self.pushButton_off)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.pushButton_clearBuffer = QtWidgets.QPushButton(self.groupBox_monitor)
        self.pushButton_clearBuffer.setMinimumSize(QtCore.QSize(160, 0))
        self.pushButton_clearBuffer.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_clearBuffer.setFont(font)
        self.pushButton_clearBuffer.setStyleSheet("")
        self.pushButton_clearBuffer.setObjectName("pushButton_clearBuffer")
        self.horizontalLayout_7.addWidget(self.pushButton_clearBuffer)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.verticalLayout.addWidget(self.groupBox_monitor)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox_parameters = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_parameters.setObjectName("groupBox_parameters")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_parameters)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox_parameters)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.spinBox_nIterations = QtWidgets.QSpinBox(self.groupBox_parameters)
        self.spinBox_nIterations.setMinimum(1)
        self.spinBox_nIterations.setMaximum(1000)
        self.spinBox_nIterations.setProperty("value", 10)
        self.spinBox_nIterations.setObjectName("spinBox_nIterations")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox_nIterations)
        self.label_2 = QtWidgets.QLabel(self.groupBox_parameters)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.spinBox_pixelCount = QtWidgets.QSpinBox(self.groupBox_parameters)
        self.spinBox_pixelCount.setMinimum(1)
        self.spinBox_pixelCount.setMaximum(1000)
        self.spinBox_pixelCount.setProperty("value", 256)
        self.spinBox_pixelCount.setObjectName("spinBox_pixelCount")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox_pixelCount)
        self.label_10 = QtWidgets.QLabel(self.groupBox_parameters)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.lineEdit_pixelSize = QtWidgets.QLineEdit(self.groupBox_parameters)
        self.lineEdit_pixelSize.setReadOnly(True)
        self.lineEdit_pixelSize.setObjectName("lineEdit_pixelSize")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_pixelSize)
        self.label_9 = QtWidgets.QLabel(self.groupBox_parameters)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.lineEdit_nModes = QtWidgets.QLineEdit(self.groupBox_parameters)
        self.lineEdit_nModes.setReadOnly(True)
        self.lineEdit_nModes.setObjectName("lineEdit_nModes")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_nModes)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.checkBox_mask = QtWidgets.QCheckBox(self.groupBox_parameters)
        self.checkBox_mask.setObjectName("checkBox_mask")
        self.horizontalLayout.addWidget(self.checkBox_mask)
        self.checkBox_background = QtWidgets.QCheckBox(self.groupBox_parameters)
        self.checkBox_background.setObjectName("checkBox_background")
        self.horizontalLayout.addWidget(self.checkBox_background)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_5.addWidget(self.groupBox_parameters)
        self.groupBox_gpuPool = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_gpuPool.setObjectName("groupBox_gpuPool")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_gpuPool)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.listWidget_gpusInUse = QtWidgets.QListWidget(self.groupBox_gpuPool)
        self.listWidget_gpusInUse.setMinimumSize(QtCore.QSize(220, 0))
        self.listWidget_gpusInUse.setMaximumSize(QtCore.QSize(220, 16777215))
        self.listWidget_gpusInUse.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.listWidget_gpusInUse.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.listWidget_gpusInUse.setObjectName("listWidget_gpusInUse")
        self.verticalLayout_3.addWidget(self.listWidget_gpusInUse)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_addGPU = QtWidgets.QPushButton(self.groupBox_gpuPool)
        self.pushButton_addGPU.setObjectName("pushButton_addGPU")
        self.horizontalLayout_2.addWidget(self.pushButton_addGPU)
        self.pushButton_removeGPU = QtWidgets.QPushButton(self.groupBox_gpuPool)
        self.pushButton_removeGPU.setObjectName("pushButton_removeGPU")
        self.horizontalLayout_2.addWidget(self.pushButton_removeGPU)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.listWidget_gpusAvailable = QtWidgets.QListWidget(self.groupBox_gpuPool)
        self.listWidget_gpusAvailable.setMinimumSize(QtCore.QSize(220, 0))
        self.listWidget_gpusAvailable.setMaximumSize(QtCore.QSize(220, 16777215))
        self.listWidget_gpusAvailable.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.listWidget_gpusAvailable.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.listWidget_gpusAvailable.setObjectName("listWidget_gpusAvailable")
        self.verticalLayout_3.addWidget(self.listWidget_gpusAvailable)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_5.addWidget(self.groupBox_gpuPool)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.groupBox_zmqSetup = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_zmqSetup.setObjectName("groupBox_zmqSetup")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBox_zmqSetup)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox_zmqSetup)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_patternsFrom = QtWidgets.QLineEdit(self.groupBox_zmqSetup)
        self.lineEdit_patternsFrom.setObjectName("lineEdit_patternsFrom")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_patternsFrom)
        self.label_7 = QtWidgets.QLabel(self.groupBox_zmqSetup)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.lineEdit_calibrationsFrom = QtWidgets.QLineEdit(self.groupBox_zmqSetup)
        self.lineEdit_calibrationsFrom.setObjectName("lineEdit_calibrationsFrom")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_calibrationsFrom)
        self.label_8 = QtWidgets.QLabel(self.groupBox_zmqSetup)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.lineEdit_resultsTo = QtWidgets.QLineEdit(self.groupBox_zmqSetup)
        self.lineEdit_resultsTo.setObjectName("lineEdit_resultsTo")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_resultsTo)
        self.verticalLayout.addWidget(self.groupBox_zmqSetup)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 504, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.label_3.setBuddy(self.lineEdit_bufferSize)
        self.label_4.setBuddy(self.lineEdit_processingTime)
        self.label.setBuddy(self.spinBox_nIterations)
        self.label_2.setBuddy(self.spinBox_pixelCount)
        self.label_10.setBuddy(self.lineEdit_pixelSize)
        self.label_9.setBuddy(self.lineEdit_nModes)
        self.label_6.setBuddy(self.lineEdit_patternsFrom)
        self.label_7.setBuddy(self.lineEdit_calibrationsFrom)
        self.label_8.setBuddy(self.lineEdit_resultsTo)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Abe\'s Extra Special Live RPI Reconstruction Service"))
        self.groupBox_monitor.setTitle(_translate("MainWindow", "Reconstruction Engine"))
        self.label_3.setText(_translate("MainWindow", "Buffer Size:"))
        self.label_4.setText(_translate("MainWindow", "Processing Time:"))
        self.pushButton_on.setText(_translate("MainWindow", "On"))
        self.pushButton_off.setText(_translate("MainWindow", "Off"))
        self.pushButton_clearBuffer.setText(_translate("MainWindow", "Clear Buffer"))
        self.groupBox_parameters.setTitle(_translate("MainWindow", "Reconstruction Parameters"))
        self.label.setText(_translate("MainWindow", "# of Iterations"))
        self.label_2.setText(_translate("MainWindow", "Pixel Count:"))
        self.label_10.setText(_translate("MainWindow", "Pixel Size:"))
        self.label_9.setText(_translate("MainWindow", "# of Modes:"))
        self.checkBox_mask.setText(_translate("MainWindow", "Mask"))
        self.checkBox_background.setText(_translate("MainWindow", "Background"))
        self.groupBox_gpuPool.setTitle(_translate("MainWindow", "GPU Pool"))
        self.listWidget_gpusInUse.setSortingEnabled(True)
        self.pushButton_addGPU.setText(_translate("MainWindow", "Add (↑)"))
        self.pushButton_removeGPU.setText(_translate("MainWindow", "Remove (↓)"))
        self.listWidget_gpusAvailable.setSortingEnabled(True)
        self.groupBox_zmqSetup.setTitle(_translate("MainWindow", "ØMQ Setup"))
        self.label_6.setText(_translate("MainWindow", "Get Patterns from:"))
        self.label_7.setText(_translate("MainWindow", "Get Calibrations from:"))
        self.label_8.setText(_translate("MainWindow", "Send Results to:"))
