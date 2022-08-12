import sys
import argparse

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#import qdarkstyle

from multiprocessing import Process, Manager

import torch as t
import zmq

from live_rpi_reconstruction_service_ui import Ui_MainWindow
from service import run_RPI_service

"""
Below we define the functionality of the main window
"""

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
        self.updateEvent = self.sharedDataManager.Event()
        self.controlInfo = self.sharedDataManager.dict()
        self.readoutInfo = self.sharedDataManager.dict()
        
    def connectSignalsSlots(self):
        self.setupOnOff()
        self.setupGPUs()

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
            
            self.controlInfo['nIterations'] = self.spinBox_nIterations.value()
            self.controlInfo['pixelCount'] = self.spinBox_pixelCount.value()
            self.controlInfo['mask'] = self.checkBox_mask.checkState()
            self.controlInfo['background'] = \
                self.checkBox_background.checkState()
            

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
                      self.controlInfo,
                      self.readoutInfo))
            self.setupReadouts()
            self.rpiService.start()

                    
        self.pushButton_on.clicked.connect(switchOn)
        self.pushButton_off.clicked.connect(switchOff)
            
        self.pushButton_clearBuffer.clicked.connect(self.clearBufferEvent.set)

    def countGPUs(self):
        in_use = self.listWidget_gpusInUse.count()
        total = in_use + self.listWidget_gpusAvailable.count()
        self.groupBox_gpuPool.setTitle('GPU Pool (%d/%d)' %
                                       (in_use, total))
        
    def setupGPUs(self):
        GPUs = [('cuda:%d; ' % i) + t.cuda.get_device_name(i) for i in range(t.cuda.device_count())]*5

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
        self.idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.readLatest)
        self.timer.start(50)

    def readLatest(self):
        if not self.rpiService.is_alive():
            self.rpiService.join()
            self.pushButton_off.click()

        elif self.updateEvent.is_set():
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
        
def main(argv=sys.argv):

    # I don't really know why, but without this the window doesn't close
    # when errors are thrown.
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook
    
    parser = argparse.ArgumentParser(description='Abe\'s Extra Special Live RPI Reconstruction Service')
    parser.add_argument('--patterns', '-p', type=str, help='ZeroMQ port to broadcast on', default='tcp://localhost:5555')
    parser.add_argument('--calibrations', '-c', type=str, help='ZeroMQ port to broadcast on', default='tcp://localhost:5557')
    parser.add_argument('--results', '-r', type=str, help='ZeroMQ port to broadcast on', default='tcp://*:5556')
    args = parser.parse_args()
    
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

