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

    CHUNK = 256

    def __init__(self):

        super().__init__()
        self.stream = self.getStream()

        rawData = self.stream.read(self.CHUNK)
        data = np.fromstring(rawData, np.int16)
        print(data)

        self.initGui()


    def update(self):

        then = datetime.now()
        t = np.linspace(0, 10, 101)
        y = np.sin(t + time.time())
        self.plotWidget.clear()
        self.plotWidget.plot(t, y)

        print('{} ms'.format((datetime.now() - then).total_seconds()))


    def initGui(self):

        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        layout = QVBoxLayout(mainWidget)
        self.plotWidget = pg.plot()
        layout.addWidget(self.plotWidget)

        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(20)
        timer.start()


    def getStream(self):

        pa = pyaudio.PyAudio()
        hostInfo = pa.get_host_api_info_by_index(0)
        nDevices = hostInfo['deviceCount']
        devices = \
            [(x, pa.get_device_info_by_host_api_device_index(0, x)) for x in range(nDevices)]
        inputDevices = list(filter(lambda x: x[1]['maxInputChannels'] > 0, devices))
        usbMicrophones = \
            list(filter(lambda x: 'USB PnP Audio Device' in x[1]['name'], inputDevices))

        # pp = pprint.PrettyPrinter()
        # for x, d in usbMicrophones:
        #     print('{} {}'.format(x, d['name']))
        #     pp.pprint(d)

        assert(len(usbMicrophones) > 0, 'No microphone detected')
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
