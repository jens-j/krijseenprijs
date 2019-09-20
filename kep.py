#!/usr/bin/env python3

import sys
import time
import pprint
import pyaudio
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, Qt
from PyQt5.uic import loadUi

# TODO:
# - spectrum scaling
# - supersampling 
# - spectrum vertical next to spectrogram 
# - audio buffering

class KrijsEenPrijs(QObject):

    TIME_AXIS_LENGTH        = 8    # s
    SAMPLE_RATE             = 48000 # Hz
    DECIMATION              = 16
    CHUNK_SIZE              = 2**11
    FFT_SIZE                = 2**11
    SPECTRUM_MAX            = 2000
    SPECTRUM_DECAY          = 5     # s
    SPECTRUM_SMOOTH         = 1
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    UPDATE_FREQ             = SAMPLE_RATE // CHUNK_SIZE

    updateSignal = pyqtSignal()

    def __init__(self):

        super(KrijsEenPrijs, self).__init__()
        self.t0 = datetime.now()
        self.u0 = datetime.now()
        self.running = True
        self.sampleCounter = 0
        self.maxSpectrum = np.zeros(self.FFT_SIZE // 2 + 1, dtype=np.int16)
        self.trailSpectrum = np.zeros(self.FFT_SIZE // 2 + 1, dtype=np.int16)
        self.spectrogram = \
            np.zeros((self.FFT_SIZE // 2 + 1, 
                      self.TIME_AXIS_SAMPLES // self.FFT_SIZE),
                     dtype=np.float32)
        self.spectrumScale = \
            np.linspace(0, self.SAMPLE_RATE / 2, self.FFT_SIZE / 2 + 1)
        self.updateSignal.connect(self.updateGui)
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.initGui()
        self.stream = self.getStream()


    def initGui(self):

        self.ui = loadUi('kep.ui')
        pg.setConfigOptions(
            imageAxisOrder='row-major', background=pg.mkColor(0x0, 0x0, 0x100, 0x24))
        
        self.ui.timePlot.setLabels(title='Time Series', left='amplitude', bottom='s')
        self.ui.timePlot.setRange(yRange=[-2**16 / 2, 2**16 / 2])

        self.ui.spectrumPlot.setLabels(title='Power Spectral Density', left='Hz', bottom='P/sqrt(Hz)')
        self.ui.spectrumPlot.getPlotItem().setLogMode(True, False)
        self.ui.spectrumPlot.getPlotItem().setRange(xRange=[-4, 4])
        
        self.spectrogramImage = pg.ImageItem()
        self.ui.spectrogramPlot.setLabels(title='Spectrogram', left='FFT bin', bottom='s')
        self.ui.spectrogramPlot.addItem(self.spectrogramImage)
        self.spectrogramImage.setLevels([0, self.SPECTRUM_MAX / 2])

        pos = np.array([0., 0.3, 0.7, 1.0])
        color = np.array([[0x0,  0x0,  0x24, 0xff], 
                          [0x17, 0x29, 0xc3, 0xff], 
                          [0xd0, 0x24, 0x12, 0xff], 
                          [0xff, 0xc4, 0x0,  0xff]], dtype=np.ubyte)

        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spectrogramImage.setLookupTable(lut)

        self.ui.startButton.clicked.connect(self.start)
        self.ui.stopButton.clicked.connect(self.stop)
        self.ui.resetButton.clicked.connect(self.reset)

        self.ui.resize(1920, 1080)
        self.ui.show()


    def updateGui(self):

        t1 = datetime.now()

        data = self.audioData[:]
        paddedData = np.concatenate([data, np.zeros(self.FFT_SIZE - self.CHUNK_SIZE)])

        t = np.linspace(-self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES / self.DECIMATION)

        spectrum = np.abs(np.fft.rfft(paddedData[-self.FFT_SIZE:]))
        spectrum = spectrum * 2 / self.FFT_SIZE
        # spectrum = spectrum / np.power(2.0, 8 * 16 - 1)
        # spectrum = (20 * np.log10(spectrum)).clip(-120)

        self.trailSpectrum -= np.array(
            [self.SPECTRUM_MAX / self.SPECTRUM_DECAY / self.UPDATE_FREQ] * len(self.maxSpectrum),
            dtype=np.int16)

        window = np.ones(self.SPECTRUM_SMOOTH) / self.SPECTRUM_SMOOTH
        smoothSpectrum = np.convolve(spectrum, window, mode='same')

        self.maxSpectrum = np.maximum(self.maxSpectrum, smoothSpectrum)
        #self.trailSpectrum = np.maximum(self.trailSpectrum, smoothSpectrum)

        self.spectrogram = self.spectrogram[:,1:]
        self.spectrogram = np.column_stack([self.spectrogram, np.float32(spectrum)])

        t2 = datetime.now()
        
        self.ui.timePlot.clear()
        self.ui.timePlot.plot(t, data[::self.DECIMATION])
        self.ui.spectrumPlot.clear()
        self.ui.spectrumPlot.plot(spectrum, self.spectrumScale)
        #self.ui.spectrumPlot.plot(self.trailSpectrum, self.spectrumScale, pen=pg.mkPen('r', style=Qt.DotLine))
        self.ui.spectrumPlot.plot(self.maxSpectrum, self.spectrumScale, 
            pen=pg.mkPen('y', style=Qt.DotLine))

        self.spectrogramImage.setImage(self.spectrogram, autoLevels=False)

        t3 = datetime.now()

        # print('{} ms calc, {} ms plot, {} ms update'.format(
        #     int((t2 - t1).total_seconds() * 1000),
        #     int((t3 - t2).total_seconds() * 1000),
        #     int((t1 - self.t0).total_seconds() * 1000)))

        self.t0 = t1


    def streamCallback(self, inputData, frameCount, timeInfo, status):

        u1 = datetime.now()

        data = np.fromstring(inputData, np.int16)

        if self.running:
            self.newAudioData = np.append(self.audioData, data)
            self.audioData = self.newAudioData[-self.TIME_AXIS_SAMPLES:]
            self.updateSignal.emit()

        # print('{} ms'.format(int((u1 - self.u0).total_seconds() * 1000)))
        self.u0 = u1

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
                       frames_per_buffer=self.FFT_SIZE,
                       input=True,
                       stream_callback=self.streamCallback)


    def start(self):
        self.running = True


    def stop(self):
        self.running = False


    def reset(self):
        self.maxSpectrum = np.zeros(self.FFT_SIZE // 2 + 1, dtype=np.int16)


def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
