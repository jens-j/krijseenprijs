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
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, QTimer

# TODO:
# - spectrum scaling
# - supersampling 
# - spectrum vertical next to spectrogram 

class KrijsEenPrijs(QWidget):

    TIME_AXIS_LENGTH        = 5    # s
    SAMPLE_RATE             = 48000 # Hz
    UPDATE_T                = 0.10 # s
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    DECIMALTION             = 8
    SPECTRUM_FFT_SIZE       = 2**13
    SPECTROGRAM_FFT_SIZE    = SAMPLE_RATE * UPDATE_T
    CHUNK_SIZE              = int(SAMPLE_RATE * UPDATE_T)

    updateSignal = pyqtSignal()

    def __init__(self):

        super().__init__()
        self.sampleCounter = 0
        self.spectrumScale = np.linspace(0, self.SAMPLE_RATE / 2, self.SPECTRUM_FFT_SIZE / 2 + 1)
        self.updateSignal.connect(self.updateGui)
        self.stream = self.getStream()
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.initGui()
        self.t0 = datetime.now()


    def initGui(self):

        self.resize(1920, 1080)
        pg.setConfigOptions(imageAxisOrder='row-major')
        layout = QVBoxLayout(self)

        self.timePlot = pg.plot()
        self.spectrumPlot = pg.plot()
        self.spectrogramPlot = pg.plot()
        self.spectrogramImage = pg.ImageItem()

        self.timePlot.setRange(yRange=[-2**16 / 2, 2**16 / 2])
        self.spectrumPlot.setRange(yRange=[0, 1000])
        self.spectrumPlot.setLogMode(x=True)
        self.spectrumPlot.set_scientific(False)
        self.spectrogramPlot.addItem(self.spectrogramImage)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.spectrogramImage)
        hist.plot.setLogMode(False, True)
        hist.setLevels(0, 400)
        hist.gradient.restoreState(
        {'mode': 'rgb',
         'ticks': [(0.5, (0, 182, 188, 255)),
                   (1.0, (246, 111, 0, 255)),
                   (0.0, (75, 0, 113, 255))]})

        layout.addWidget(self.timePlot)
        layout.addWidget(self.spectrumPlot)
        layout.addWidget(self.spectrogramPlot)

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.updateGui)
        # self.timer.start(self.UPDATE_T * 1000)
        # self.timer.start()


    def updateGui(self):

        t1 = datetime.now()
        data = self.audioData[:]

        print(int((datetime.now() - self.t0).total_seconds() * 1000))

        t = np.linspace(-self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES / self.DECIMALTION)
        spectrum = np.abs(np.fft.rfft(data[-self.SPECTRUM_FFT_SIZE:])) / self.SPECTRUM_FFT_SIZE
        f, _, spectrogram = signal.spectrogram(
            data, self.SAMPLE_RATE, nperseg=self.SPECTROGRAM_FFT_SIZE)
        spectrogram = spectrogram / self.SPECTRUM_FFT_SIZE

        print(np.max(spectrogram))
        
        self.timePlot.clear()
        self.timePlot.plot(t, data[::self.DECIMALTION])
        self.spectrumPlot.clear()
        self.spectrumPlot.plot(self.spectrumScale, spectrum)
        self.spectrogramImage.setImage(spectrogram)

        self.t0 = t1


    def streamCallback(self, inputData, frameCount, timeInfo, status):

        data = np.fromstring(inputData, np.int16)
        self.newAudioData = np.append(self.audioData, data)
        self.audioData = self.newAudioData[-self.TIME_AXIS_SAMPLES:]

        # self.sampleCounter += self.CHUNK_SIZE
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
                       frames_per_buffer=self.CHUNK_SIZE,
                       input=True,
                       stream_callback=self.streamCallback)



def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    kep.show()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
