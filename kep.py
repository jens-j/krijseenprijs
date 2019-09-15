#!/usr/bin/env python3

import sys
import time
import pprint
import pyaudio
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer



class KrijsEenPrijs(QMainWindow):

    TIME_AXIS_LENGTH    = 10    # s
    SAMPLE_RATE         = 48000 # Hz
    UPDATE_T            = 0.05  # s
    TIME_AXIS_SAMPLES   = SAMPLE_RATE * TIME_AXIS_LENGTH
    DECIMALTION         = 8

    def __init__(self):

        super().__init__()
        self.t0 = datetime.now()
        self.stream = self.getStream()
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.initGui()


    def update(self):

        t1 = datetime.now()
        print((datetime.now() - self.t0).total_seconds() * 1000)
        t = np.linspace(0, self.TIME_AXIS_LENGTH, self.TIME_AXIS_SAMPLES / self.DECIMALTION)
        self.plotWidget.clear()
        self.plotWidget.plot(t, self.audioData[::self.DECIMALTION])
        self.t0 = t1


    def initGui(self):

        self.resize(800, 600)
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        layout = QVBoxLayout(mainWidget)
        self.plotWidget = pg.plot()
        self.plotWidget.setRange(yRange=[-2**16/2, 2**16/2])
        layout.addWidget(self.plotWidget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_T * 1000)
        self.timer.start()


    def streamCallback(self, inputData, frameCount, timeInfo, status):

        data = np.fromstring(inputData, np.int16)
        self.newAudioData = np.append(self.audioData, data)
        self.audioData = self.newAudioData[-self.TIME_AXIS_SAMPLES:]

        return (inputData, pyaudio.paContinue)


    def getStream(self):

        pa = pyaudio.PyAudio()
        hostInfo = pa.get_host_api_info_by_index(0)
        nDevices = hostInfo['deviceCount']
        devices = \
            [(x, pa.get_device_info_by_host_api_device_index(0, x)) for x in range(nDevices)]
        inputDevices = list(filter(lambda x: x[1]['maxInputChannels'] > 0, devices))
        usbMicrophones = \
            list(filter(lambda x: 'USB PnP Audio Device' in x[1]['name'], inputDevices))

        assert len(usbMicrophones) > 0, 'No microphone detected'
        mic = usbMicrophones[0][1]

        return pa.open(format=pyaudio.paInt16,
                       input_device_index=mic['index'],
                       channels=mic['maxInputChannels'],
                       rate=int(mic['defaultSampleRate']), 
                       input=True,
                       stream_callback=self.streamCallback)



def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    kep.show()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
