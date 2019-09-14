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

# CHUNK = 256

# pp = pprint.PrettyPrinter()
# pa = pyaudio.PyAudio()

# hostInfo = pa.get_host_api_info_by_index(0)
# nDevices = hostInfo['deviceCount']

# devices = [(x, pa.get_device_info_by_host_api_device_index(0, x)) for x in range(nDevices)]
# inputDevices = list(filter(lambda x: x[1]['maxInputChannels'] > 0, devices))
# usbMicrophones = list(filter(lambda x: 'USB PnP Audio Device' in x[1]['name'], inputDevices))

# for x, d in usbMicrophones:
#     print('{} {}'.format(x, d['name']))
#     pp.pprint(d)

# mic = usbMicrophones[0][1]

# stream = pa.open(format=pyaudio.paInt16,
#                  input_device_index=mic['index'],
#                  channels=mic['maxInputChannels'],
#                  rate=int(mic['defaultSampleRate']), 
#                  input=True,
#                  frames_per_buffer=CHUNK)

# rawData = stream.read(CHUNK)
# data = np.fromstring(rawData, np.int16)
# print(data)


class KrijsEenPrijs(QMainWindow):

    TIME_AXIS_LENGTH    = 1 # s
    SAMPLE_RATE         = 48000 # Hz
    UPDATE_T            = 0.02  # s
    CHUNK               = int(SAMPLE_RATE * UPDATE_T)
    TIME_AXIS_SAMPLES   = SAMPLE_RATE * TIME_AXIS_LENGTH

    def __init__(self):

        super().__init__()
        self.stream = self.getStream()
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.initGui()
        print(self.CHUNK)


    def update(self):

        # then = datetime.now()

        rawData = self.stream.read(self.CHUNK, exception_on_overflow=False)
        data = np.fromstring(rawData, np.int16)
        self.audioData = np.append(self.audioData, data)[-self.TIME_AXIS_SAMPLES:]
        t = np.linspace(0, 1, self.TIME_AXIS_SAMPLES)
        self.plotWidget.clear()
        self.plotWidget.plot(t, self.audioData)

        # print('{} ms'.format((datetime.now() - then).total_seconds()))



    def initGui(self):

        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        layout = QVBoxLayout(mainWidget)
        self.plotWidget = pg.plot()
        layout.addWidget(self.plotWidget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_T * 1000)
        self.timer.start()


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
                       frames_per_buffer=self.CHUNK)


def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    kep.show()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
