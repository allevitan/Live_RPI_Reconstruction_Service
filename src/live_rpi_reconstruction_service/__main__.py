import sys
import argparse

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#import qdarkstyle

import multiprocessing
from multiprocessing import Process, Manager

import torch as t
import zmq

from live_rpi_reconstruction_service.live_rpi_reconstruction_service_ui \
    import Ui_MainWindow
from live_rpi_reconstruction_service.service import run_RPI_service

# This is apparently the best-practice way to load config files from within
# the package
import importlib.resources
import json

"""
Below we define the functionality of the main window
"""

# TODO: Right now if the number of iterations drops, it can output a later
# result before an earlier one because the later one took less time. So I
# should make an output buffer than ensures the ordering is correct.

# TODO: Make it actually use more than one probe mode for the reconstruction!

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, sharedDataManager, parent=None,
                 patternsPort='', calibrationsPort='',
                 resultsPort=''):
        super().__init__(parent)
        self.setupUi(self)

        self.setupZMQ(patternsPort, calibrationsPort, resultsPort)
        self.sharedDataManager = sharedDataManager
        self.setupSharedObjects()
        self.connectSignalsSlots()

    def setupZMQ(self, patternsPort, calibrationsPort, resultsPort):
        self.lineEdit_patternsFrom.setText(patternsPort)
        self.lineEdit_calibrationsFrom.setText(calibrationsPort)
        self.lineEdit_resultsTo.setText(resultsPort)

    def setupSharedObjects(self):
        self.stopEvent = self.sharedDataManager.Event()
        self.clearBufferEvent = self.sharedDataManager.Event()
        self.calibrationAcquiredEvent = self.sharedDataManager.Event()
        self.updateEvent = self.sharedDataManager.Event()
        self.controlInfo = self.sharedDataManager.dict()
        self.readoutInfo = self.sharedDataManager.dict()
        
    def connectSignalsSlots(self):
        self.setupOnOff()
        self.setupGPUs()
        self.setupReconstructionParameters()

    def setupOnOff(self):

        def switchOff():
            if hasattr(self, 'rpiService'):
                self.stopEvent.set()
                self.rpiService.join()
                self.timer.stop()

            self.pushButton_on.setChecked(False)
            self.pushButton_on.setStyleSheet('')
            self.pushButton_off.setChecked(True)
            self.pushButton_off.setStyleSheet('color: red;')
            
            self.groupBox_zmqSetup.setDisabled(False)
            self.groupBox_gpuPool.setDisabled(False)

            self.removeCalibration()

        def switchOn():
            if not self.pushButton_on.isChecked():
                # This means it was already clicked
                self.pushButton_on.setChecked(True)
                return

            if self.listWidget_gpusInUse.count() == 0:
                msg = QMessageBox()
                msg.setText('You need to add a GPU before you can start the service!')
                msg.exec()
                self.pushButton_on.setChecked(False)
                return
            
            self.pushButton_on.setChecked(True)
            self.pushButton_on.setStyleSheet('color: green;')
            self.pushButton_off.setChecked(False)
            self.pushButton_off.setStyleSheet('')

            self.groupBox_zmqSetup.setDisabled(True)
            self.groupBox_gpuPool.setDisabled(True)

            self.label_calibrationState.setStyleSheet('color: red;')
            self.label_calibrationState.setText('Waiting for Calibration...')
            
            self.controlInfo['nIterations'] = self.spinBox_nIterations.value()
            self.controlInfo['pixelCount'] = self.spinBox_pixelCount.value()
            # The checkState() output is some special pyqt object which can't
            # be pickled, so it can't be sent over the multiprocessing manager
            self.controlInfo['mask'] = bool(self.checkBox_mask.checkState())
            self.controlInfo['background'] = \
                bool(self.checkBox_background.checkState())
            

            self.controlInfo['patternsPort'] = \
                self.lineEdit_patternsFrom.text()
            self.controlInfo['calibrationsPort'] = \
                self.lineEdit_calibrationsFrom.text()
            self.controlInfo['resultsPort'] = \
                self.lineEdit_resultsTo.text()

            GPU_strings = [self.listWidget_gpusInUse.item(idx).text()
                           for idx
                           in range(self.listWidget_gpusInUse.count())]
            GPUs = [string.split(';')[0] for string in GPU_strings]
                    

            self.controlInfo['GPUs'] = GPUs
            
            self.readoutInfo['bufferSize'] = 0
            self.readoutInfo['calculationTime'] = 0
            self.readoutInfo['nModes'] = 0
            self.readoutInfo['basis'] = None

            self.stopEvent.clear()

            self.rpiService = Process(
                daemon=True,
                target=run_RPI_service,
                args=(self.stopEvent,
                      self.clearBufferEvent,
                      self.updateEvent,
                      self.calibrationAcquiredEvent,
                      self.controlInfo,
                      self.readoutInfo))
            self.setupReadouts()
            #self.rpiService.start()
            self.rpiService.start()

                    
        self.pushButton_on.clicked.connect(switchOn)
        self.pushButton_off.clicked.connect(switchOff)
            
        self.pushButton_clearBuffer.clicked.connect(self.clearBufferEvent.set)


    def acquireCalibration(self):
        self.label_calibrationState.setStyleSheet('color: green;')
        self.label_calibrationState.setText('Calibration Acquired!')
        
        self.label_pixelSize.setDisabled(False)
        self.lineEdit_pixelSize.setDisabled(False)

        pixelSize = self.readoutInfo['FOV'] / self.spinBox_pixelCount.value()
        self.lineEdit_pixelSize.setText('%0.2f nm' % (pixelSize * 1e9))

        self.label_nModes.setDisabled(False)
        self.lineEdit_nModes.setDisabled(False)

        self.lineEdit_nModes.setText('%d' % self.readoutInfo['nModes'])

        self.label_oversampling.setDisabled(False)
        self.lineEdit_oversampling.setDisabled(False)

        self.lineEdit_oversampling.setText('%d' % self.readoutInfo['oversampling'])

        if self.readoutInfo['hasMask']:
            self.checkBox_mask.setDisabled(False)
        else:
            self.checkBox_mask.setDisabled(True)
        if self.readoutInfo['hasBackground']:
            self.checkBox_background.setDisabled(False)
        else:
            self.checkBox_background.setDisabled(True)

    def setupReconstructionParameters(self):
        self.removeCalibration()

        def updateNIterations(nIterations):
            self.controlInfo['nIterations'] = nIterations
        
        self.spinBox_nIterations.valueChanged.connect(updateNIterations)

        def updatePixelCount(pixelCount):
            self.controlInfo['pixelCount'] = pixelCount
            if 'FOV' in self.readoutInfo:
                pixelSize = (self.readoutInfo['FOV']
                             / self.spinBox_pixelCount.value())
                self.lineEdit_pixelSize.setText('%0.2f nm' % (pixelSize * 1e9))

        self.spinBox_pixelCount.valueChanged.connect(updatePixelCount)
        
        
    def removeCalibration(self):
        self.label_calibrationState.setStyleSheet('')
        self.label_calibrationState.setText('Calibration Dependent Info:')
        self.label_pixelSize.setDisabled(True)
        self.lineEdit_pixelSize.setDisabled(True)
        self.label_nModes.setDisabled(True)
        self.lineEdit_nModes.setDisabled(True)
        self.label_oversampling.setDisabled(True)
        self.lineEdit_oversampling.setDisabled(True)
        if 'FOV' in self.readoutInfo:
            self.readoutInfo.pop('FOV')

    
    def countGPUs(self):
        in_use = self.listWidget_gpusInUse.count()
        total = in_use + self.listWidget_gpusAvailable.count()
        self.groupBox_gpuPool.setTitle('GPU Pool (%d/%d)' %
                                       (in_use, total))
        
    def setupGPUs(self):
        GPUs = [('cuda:%d; ' % i) + t.cuda.get_device_name(i) for i in range(t.cuda.device_count())]

        self.listWidget_gpusAvailable.addItems(GPUs)

        def addToPool():
            selection = self.listWidget_gpusAvailable.currentRow()
            if selection == -1: #No selection
                return
            item = self.listWidget_gpusAvailable.takeItem(selection)
            self.listWidget_gpusInUse.addItem(item.text())
            self.countGPUs()

        def removeFromPool():
            selection = self.listWidget_gpusInUse.currentRow()
            if selection == -1: #No selection
                return
            item = self.listWidget_gpusInUse.takeItem(selection)
            self.listWidget_gpusAvailable.addItem(item.text())
            self.countGPUs()

        self.pushButton_addGPU.clicked.connect(addToPool)
        self.pushButton_removeGPU.clicked.connect(removeFromPool)

        self.countGPUs()
        
        self.listWidget_gpusInUse.model().rowsRemoved.connect(self.countGPUs)
        self.listWidget_gpusAvailable.model().rowsRemoved.connect(
            self.countGPUs)
    
    def setupReadouts(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.readLatest)
        self.timer.start(50)

    def readLatest(self):
        if not self.rpiService.is_alive():
            self.rpiService.join()
            self.pushButton_off.click()

        if self.updateEvent.is_set():
            self.updateEvent.clear()
            
            bufferSize = self.readoutInfo['bufferSize']
            self.lineEdit_bufferSize.setText(str(bufferSize))
            if bufferSize < 10:
                self.lineEdit_bufferSize.setStyleSheet('color: green;')
            elif bufferSize < 100:
                self.lineEdit_bufferSize.setStyleSheet('color: orange;')
            else:
                self.lineEdit_bufferSize.setStyleSheet('color: red;')

            calculationTime = self.readoutInfo['processingTime']
            self.lineEdit_processingTime.setText(
                '%0.2f ms' % (calculationTime*1000))

        if self.calibrationAcquiredEvent.is_set():
            self.calibrationAcquiredEvent.clear()
            self.acquireCalibration()
        
def main(argv=sys.argv):

    # This is needed to allow CUDA to be initialized in both the main thread
    # and the subprocesses
    multiprocessing.set_start_method('spawn')
    
    # I don't really know why, but without this the window doesn't close
    # when errors are thrown.
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook
    
    parser = argparse.ArgumentParser(description='Abe\'s Extra Special Live RPI Reconstruction Service')
    parser.add_argument('--patterns', '-p', type=str, help='ZeroMQ port to listen for patterns on', default='')
    parser.add_argument('--calibrations', '-c', type=str, help='ZeroMQ port to broadcast on', default='')
    parser.add_argument('--results', '-r', type=str, help='ZeroMQ port to broadcast on', default='')
    args = parser.parse_args()


    package_root = importlib.resources.files('live_rpi_reconstruction_service')
    # This loads the default configuration first. This file is managed by
    # git and should not be edited by a user
    config = json.loads(package_root.joinpath('defaults.json').read_text())\

    # And now, if the user has installed an optional config file, we allow it
    # to override what is in defaults.json
    config_file_path = package_root.joinpath('config.json')

    # not sure if this works with zipped packages
    if config_file_path.exists():
        config.update(json.loads(config_file_path.read_text()))

    # Now we set the defaults, if they haven't been set yet
    if args.patterns == '':
        args.patterns = config['data_subscription_port']
    if args.calibrations == '':
        args.calibrations = config['calibration_subscription_port']
    if args.results == '':
        args.results = config['rpi_frame_publish_port']
        
    
    app = QApplication(sys.argv)
    app.setApplicationName('RPI Reconstruction Service')
    #app.setStyleSheet(qdarkstyle.load_stylesheet())

    with Manager() as sharedDataManager:
        win = Window(sharedDataManager,
                     patternsPort=args.patterns,
                     calibrationsPort=args.calibrations,
                     resultsPort=args.results)
        win.show()
        return app.exec()

if __name__ == '__main__':
    sys.exit(main())

