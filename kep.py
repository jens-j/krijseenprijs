#!/usr/bin/env python3

import sys
import time
import pprint
import pyaudio
from datetime import datetime
from collections import deque

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, Qt
from PyQt5.QtGui import QFont
from PyQt5.uic import loadUi

# TODO:
# - log spectrogram
# - plot ticks
# - grid
# - pdf

class KrijsEenPrijs(QObject):

    TIME_AXIS_LENGTH        = 8    # s
    SAMPLE_RATE             = 48000 # Hz
    DECIMATION              = 32
    CHUNK_SIZE              = 2**11
    CHUNK_DIVIDE            = 2
    FFT_SIZE                = 2**12
    FFT_RESOLUTION          = SAMPLE_RATE // FFT_SIZE
    SPECTRUM_MIN            = -160
    SPECTRUM_MAX            = 0
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    UPDATE_FREQ             = SAMPLE_RATE // CHUNK_SIZE

    updateSignal = pyqtSignal()

    def __init__(self):

        super(KrijsEenPrijs, self).__init__()

        self.deque = deque()
        self.running = True
        self.t0 = datetime.now()
        self.maxFrequency = 0
        self.maxPower = self.SPECTRUM_MIN
        self.spectrumScale = np.linspace(0, self.SAMPLE_RATE / 2, self.FFT_SIZE / 2 + 1)
        self.timeAxis = np.linspace(
            -self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES / self.DECIMATION)
        self.logScaleSpectrogram = (np.log(range(1, self.FFT_SIZE // 2 + 2)) 
            / np.log(self.FFT_SIZE // 2 + 2) * self.FFT_SIZE // 2 + 1)
        
        self.initPlotData()
        self.initGui()
        self.updateSignal.connect(self.updateGui)
        self.stream = self.getStream()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateGui)
        self.timer.start(self.SAMPLE_RATE / self.CHUNK_SIZE)
        self.timer.start()


    def initPlotData(self):

        self.maxPower = self.SPECTRUM_MIN
        self.audioData = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.maxSpectrum = np.ones(self.FFT_SIZE // 2 + 1, dtype=np.int16) * self.SPECTRUM_MIN
        self.spectrogram = np.ones((self.FFT_SIZE // 2 + 1, 
                             self.TIME_AXIS_SAMPLES // (self.CHUNK_SIZE // self.CHUNK_DIVIDE)),
                            dtype=np.float32) * self.SPECTRUM_MIN


    def initGui(self):

        self.ui = loadUi('kep.ui')
        pg.setConfigOptions(
            imageAxisOrder='row-major', background=pg.mkColor(0x0, 0x0, 0x100, 0x24))

        font = QFont()
        font.setPixelSize(14)
        
        labelStyle = {'color': '#FFF', 'font-size': '20px'}
        self.ui.timePlot.setLabel('left', 'amplitude', **labelStyle)
        #self.ui.timePlot.setLabels(title='Amplitude', left='amplitude', bottom='s')
        self.ui.timePlot.setRange(yRange=[-2**15, 2**15])
        self.ui.timePlot.getAxis('left').tickFont = font
        self.timeCurve = self.ui.timePlot.plot()

        self.ui.spectrumPlot.setLabels(
            title='Power Spectral Density', left='Hz', bottom='dBFS')
        self.ui.spectrumPlot.getPlotItem().setLogMode(False, True)
        self.ui.spectrumPlot.getPlotItem().setRange(xRange=[-160, 1])
        self.ui.spectrumPlot.getAxis('bottom').setTicks(
            [[(x, str(x)) for x in range(-160, 20, 20)]])
        # self.ui.spectrumPlot.getAxis('left').setTicks(
        #     [[(x, str(x)) for x in [0, 10, 100, 1000, 10000]]])
        self.spectrumCurve = self.ui.spectrumPlot.plot()
        self.spectrumMaxCurve = self.ui.spectrumPlot.plot()
        
        self.spectrogramImage = pg.ImageItem()
        self.ui.spectrogramPlot.setLabels(title='Spectrogram', left='FFT bin', bottom='s')

        self.ui.spectrogramPlot.addItem(self.spectrogramImage)
        self.spectrogramImage.setLevels([-160, 0])

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


    def getSpectrum(self, data):

        data = data * np.hanning(len(data))
        data = np.concatenate([data, np.zeros(self.FFT_SIZE - len(data))])

        spectrum = np.fft.rfft(data[-self.FFT_SIZE:])
        spectrum = np.abs(spectrum) * 2 / self.FFT_SIZE
        spectrum[spectrum == 0] = 1E-6
        spectrum = 10 * np.log(spectrum / 2**15)

        return spectrum


    def updateGui(self):

        t1 = datetime.now()

        if len(self.deque) == 0:
            print('no data in queue')
            return

        newData = self.deque.popleft()
        self.audioData = np.append(self.audioData, newData)
        self.audioData = self.audioData[-self.TIME_AXIS_SAMPLES:]

        # process two spectra for every GUI update
        for i in range(self.CHUNK_DIVIDE):

            length = self.CHUNK_SIZE // self.CHUNK_DIVIDE
            x = (-self.CHUNK_DIVIDE + i) * length 
            y = (-self.CHUNK_DIVIDE + 1 + i) * length

            fftData = newData[x:] if y == 0 else newData[x:y]
            spectrum = self.getSpectrum(fftData)
            self.maxSpectrum = np.maximum(self.maxSpectrum, spectrum)
            self.spectrogram = self.spectrogram[:,1:]
            self.spectrogram = np.column_stack([self.spectrogram, np.float32(spectrum)])


        maxPower = np.amax(spectrum)
        if maxPower > self.maxPower:
            self.maxPower = maxPower
            self.maxFrequency = np.where(spectrum == maxPower)[0][0] * self.FFT_RESOLUTION

        t2 = datetime.now()
        
        self.ui.lblPower.setText('{:.2f} dBFS'.format(self.maxPower))
        self.ui.lblFrequency.setText('{} Hz'.format(self.maxFrequency))

        self.spectrogramImage.setImage(self.spectrogram, autoLevels=False)
        self.timeCurve.setData(self.timeAxis, self.audioData[::self.DECIMATION])
        self.spectrumCurve.setData(spectrum, self.spectrumScale)
        self.spectrumMaxCurve.setData(self.maxSpectrum, self.spectrumScale, 
            pen=pg.mkPen('r'))#, style=Qt.DotLine))

        t3 = datetime.now()

        # print('{} ms calc, {} ms plot, {} ms update'.format(
        #     int((t2 - t1).total_seconds() * 1000),
        #     int((t3 - t2).total_seconds() * 1000),
        #     int((t1 - self.t0).total_seconds() * 1000)))

        self.t0 = t1


    def streamCallback(self, inputData, frameCount, timeInfo, status):

        data = np.fromstring(inputData, np.int16)

        if self.running:
            self.deque.append(data)
            #self.updateSignal.emit()

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
                       channels=1,
                       rate=self.SAMPLE_RATE, 
                       frames_per_buffer=self.CHUNK_SIZE,
                       input=True,
                       stream_callback=self.streamCallback)


    def start(self):
        self.running = True


    def stop(self):
        self.running = False


    def reset(self):
        self.initPlotData()
        self.deque.append(np.zeros(self.CHUNK_SIZE))
        self.updateSignal.emit()

def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
