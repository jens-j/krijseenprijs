#!/usr/bin/env python3

import sys
import time
import pprint
import pyaudio
from datetime import datetime

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import pyqtSignal, QTimer

# TODO:
# - spectrum scaling
# - supersampling 
# - spectrum vertical next to spectrogram 
# - audio buffering

class KrijsEenPrijs(QWidget):

    TIME_AXIS_LENGTH        = 10    # s
    SAMPLE_RATE             = 48000 # Hz
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    DECIMATION              = 32
    SPECTRUM_FFT_SIZE       = 2**10
    SPECTRUM_MAX            = 3000

    updateSignal = pyqtSignal()

    def __init__(self):

        super().__init__()
        self.sampleCounter = 0
        self.spectrogram = \
            np.zeros((self.SPECTRUM_FFT_SIZE // 2 + 1, 
                      self.TIME_AXIS_SAMPLES // self.SPECTRUM_FFT_SIZE),
                     dtype=np.float32)
        self.spectrumScale = \
            np.linspace(0, self.SAMPLE_RATE / 2, self.SPECTRUM_FFT_SIZE / 2 + 1)
        self.updateSignal.connect(self.updateGui)
        self.stream = self.getStream()
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.initGui()
        self.t0 = datetime.now()


    def initGui(self):

        self.resize(1920, 1080)
        pg.setConfigOptions(imageAxisOrder='row-major')

        #plotLayout = QVBoxLayout(self)
        layout = QGridLayout(self)
        spectrogramLayout = QHBoxLayout(self)

        self.timePlot = pg.PlotWidget()
        self.spectrumPlot = pg.PlotWidget()
        self.spectrogramPlot = pg.PlotWidget()
        self.spectrogramImage = pg.ImageItem()

        self.timePlot.setRange(yRange=[-2**16 / 2, 2**16 / 2])
        #self.spectrumPlot.setLogMode(x=True)
        #self.spectrogramPlot.setLogMode(y=True)
        self.spectrogramPlot.addItem(self.spectrogramImage)
        self.spectrogramImage.setLevels([0, self.SPECTRUM_MAX])

        layout.addWidget(self.timePlot, 0, 0)
        layout.addWidget(self.spectrumPlot, 1, 0)
        layout.addWidget(self.spectrogramPlot, 2, 0)


    def updateGui(self):

        t1 = datetime.now()
        data = self.audioData[:]

        print('{} ms'.format(int((datetime.now() - self.t0).total_seconds() * 1000)))

        t = np.linspace(-self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES / self.DECIMATION)

        spectrum = np.abs(np.fft.rfft(data[-self.SPECTRUM_FFT_SIZE:]))
        spectrum = spectrum * 2 / self.SPECTRUM_FFT_SIZE
        # spectrum = spectrum / np.power(2.0, 8 * 16 - 1)
        # spectrum = (20 * np.log10(spectrum)).clip(-120)

        self.spectrogram = self.spectrogram[:,1:]
        self.spectrogram = np.column_stack([self.spectrogram, np.float32(spectrum)])
        
        self.timePlot.clear()
        self.timePlot.plot(t, data[::self.DECIMATION])
        self.spectrumPlot.clear()
        self.spectrumPlot.plot(self.spectrumScale, spectrum)
        self.spectrumPlot.setRange(yRange=[0, self.SPECTRUM_MAX])

        self.spectrogramImage.setImage(self.spectrogram, autoLevels=False)
        #self.histogram.setImageItem(self.spectrogramImage)

        self.t0 = t1


    def streamCallback(self, inputData, frameCount, timeInfo, status):

        data = np.fromstring(inputData, np.int16)
        self.newAudioData = np.append(self.audioData, data)
        self.audioData = self.newAudioData[-self.TIME_AXIS_SAMPLES:]

        # self.sampleCounter += self.SPECTRUM_FFT_SIZE
        # if self.sampleCounter >= self.SAMPLE_RATE * self.UPDATE_T:
        #     self.sampleCounter -= self.SAMPLE_RATE * self.UPDATE_T
        self.updateSignal.emit()
        #     print('emit')

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
                       frames_per_buffer=self.SPECTRUM_FFT_SIZE,
                       input=True,
                       stream_callback=self.streamCallback)



def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    kep.show()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
