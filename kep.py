#!/usr/bin/env python3

import sys
import time
import pprint
import pyaudio
import yaml
from datetime import datetime
from collections import deque

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import interp1d
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QSpacerItem
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, Qt
from PyQt5.QtGui import QFont, QSizePolicy, QFont, QPixmap
from PyQt5.uic import loadUi

# TODO:
# - log spectrogram
# - plot ticks
# - grid
# - pdf
# - same name popup
# - overlapping FFT windows
# - calibration
# - save wav
# - speed issues


class KrijsEenPrijs(QObject):

    TIME_AXIS_LENGTH        = 8    # s
    SAMPLE_RATE             = 48000 # Hz
    DECIMATION              = 32
    CHUNK_SIZE              = 2**11
    FFT_RATE                = 2**10
    FFT_SIZE                = 2**12
    LONG_FFT_SIZE           = 2**13
    FFT_RESOLUTION          = SAMPLE_RATE // FFT_SIZE
    LONG_FFT_RESOLUTION     = SAMPLE_RATE // LONG_FFT_SIZE
    SPECTRUM_MIN            = -240
    SPECTRUM_MAX            = 0
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    UPDATE_FREQ             = SAMPLE_RATE // CHUNK_SIZE

    updatePlotsSignal = pyqtSignal()
    updateScoresSignal = pyqtSignal()

    def __init__(self):

        super(KrijsEenPrijs, self).__init__()

        self.pp = pprint.PrettyPrinter()
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

        try:
            with open('./globalscores.yaml') as f:
                self.globalScores = yaml.load(f)
            if self.globalScores == None:
                self.globalScores = {}
        except:
            print('could not load global highscores')
            self.globalScores = {}

        try:
            with open('./localscores.yaml') as f:
                self.localScores = yaml.load(f)
            if self.localScores == None:
                self.localScores = {}
        except:
            print('could not load local highscores')
            self.localScores = {}
        
        self.initPlotData()
        self.initPlotsGui()
        self.initScoresGui()
        self.updatePlotsSignal.connect(self.updatePlots)
        self.updateScoresSignal.connect(self.updateScores)
        self.stream = self.getStream()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateData)
        self.updateScoresSignal.emit()
        self.timer.start(self.SAMPLE_RATE / self.CHUNK_SIZE)
        self.timer.start()

        print(self.spectrumScale)


    def initPlotData(self):

        self.maxPower    = self.SPECTRUM_MIN
        self.audioData   = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.spectrum    = np.ones(self.FFT_SIZE // 2 + 1, dtype=np.int16) * self.SPECTRUM_MIN
        self.maxSpectrum = np.ones(self.FFT_SIZE // 2 + 1, dtype=np.int16) * self.SPECTRUM_MIN
        self.spectrogram = np.ones((self.FFT_SIZE // 2 + 1, 
                            self.TIME_AXIS_SAMPLES // self.FFT_RATE),
                            dtype=np.float32) * self.SPECTRUM_MIN


    def initPlotsGui(self):
        
        pg.setConfigOptions(
            imageAxisOrder='row-major', background=pg.mkColor(0x0, 0x0, 0x100, 0x24))

        self.plots = loadUi('plots.ui')

        self.font = QFont()
        self.font.setPixelSize(14)
        self.labelStyle = {'color': '#FFF', 'font-size': '16px'}
        self.titleStyle = {'color': '#FFF', 'font-size': '40px'}

        self.plots.labelLogo.setPixmap(QPixmap('sron_small.png'))

        self.plots.timePlot.setTitle('Microphone Signal', **self.titleStyle)
        self.plots.timePlot.setLabel('left', 'amplitude', **self.labelStyle)
        self.plots.timePlot.setLabel('bottom', 'time (s)', **self.labelStyle)
        self.plots.timePlot.getAxis('left').tickFont = self.font
        self.plots.timePlot.getAxis('bottom').tickFont = self.font
        self.plots.timePlot.setRange(yRange=[-2**15, 2**15])
        
        self.plots.spectrumPlot.setTitle(title='Power Spectral Density', **self.titleStyle)
        self.plots.spectrumPlot.setLabel('left', 'frequency (Hz)', **self.labelStyle)
        self.plots.spectrumPlot.setLabel(
            'bottom', 'power spectral density (dBFS)', **self.labelStyle)
        self.plots.spectrumPlot.getAxis('left').tickFont = self.font
        self.plots.spectrumPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrumPlot.getPlotItem().setLogMode(False, True)
        self.plots.spectrumPlot.getPlotItem().setRange(
            xRange=[self.SPECTRUM_MIN, self.SPECTRUM_MAX])
        self.plots.spectrumPlot.getAxis('bottom').setTicks(
            [[(x, str(x)) for x in range(self.SPECTRUM_MIN, self.SPECTRUM_MAX + 20, 20)]])
        # self.plots.spectrumPlot.getAxis('left').setTicks(
        #     [[(x, str(x)) for x in [10, 100, 1000, 10000]]])
        
        self.timeCurve = self.plots.timePlot.plot()
        self.spectrumCurve = self.plots.spectrumPlot.plot()
        self.spectrumMaxCurve = self.plots.spectrumPlot.plot()
        
        self.spectrogramImage = pg.ImageItem()
        self.plots.spectrogramPlot.setTitle('Spectrogram', **self.titleStyle)
        self.plots.spectrogramPlot.setLabel('left', 'FFT bin', **self.labelStyle)
        self.plots.spectrogramPlot.setLabel('bottom', 'time (s)', **self.labelStyle)
        self.plots.spectrogramPlot.getAxis('left').tickFont = self.font
        self.plots.spectrogramPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrogramPlot.addItem(self.spectrogramImage)
        self.spectrogramImage.setLevels([-160, 0])

        pos = np.array([0., 0.3, 0.7, 1.0])
        color = np.array([[0x0,  0x0,  0x24, 0xff], 
                          [0x17, 0x29, 0xc3, 0xff], 
                          [0xd0, 0x24, 0x12, 0xff], 
                          [0xff, 0xc4, 0x0,  0xff]], dtype=np.ubyte)

        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spectrogramImage.setLookupTable(lut)

        self.plots.startButton.clicked.connect(self.start)
        self.plots.stopButton.clicked.connect(self.stop)
        self.plots.saveButton.clicked.connect(self.save)
        self.plots.resetButton.clicked.connect(self.reset)

        self.plots.setWindowTitle('Plots')
        self.plots.resize(1920, 1080)
        self.plots.show()


    def initScoresGui(self):

        self.scores = loadUi('scores.ui')

        self.scores.powerHistogram.setTitle('Power Score Distribution', **self.titleStyle)
        self.scores.powerHistogram.setLabel('left', '# of people', **self.labelStyle)
        self.scores.powerHistogram.setLabel('bottom', 'Power (dBFS)', **self.labelStyle)
        self.scores.powerHistogram.getAxis('left').tickFont = self.font
        self.scores.powerHistogram.getAxis('bottom').tickFont = self.font

        self.scores.frequencyHistogram.setTitle('Main Frequency Distribution', **self.titleStyle)
        self.scores.frequencyHistogram.setLabel('left', '# of people', **self.labelStyle)
        self.scores.frequencyHistogram.setLabel('bottom', 'frequency (Hz)', **self.labelStyle)
        self.scores.frequencyHistogram.getAxis('left').tickFont = self.font
        self.scores.frequencyHistogram.getAxis('bottom').tickFont = self.font

        powerScores = [x[0] for x in self.globalScores.values()]
        y, x = np.histogram(powerScores, bins=np.linspace(-100, 0, 21))
        self.scores.powerHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

        frequencyScores = [x[1] for x in self.globalScores.values()]
        y, x = np.histogram(
            frequencyScores, bins=np.linspace(0, self.SAMPLE_RATE // 4, self.FFT_SIZE // 4 + 1))
        self.scores.frequencyHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

        self.scores.setWindowTitle('High Scores')
        self.scores.resize(1920, 1080)
        self.scores.show()


    def getSpectrum(self, data):

        SIZE = len(data)

        data = data * np.hanning(SIZE) 

        spectrum = np.fft.rfft(data[-SIZE:])
        spectrum = np.abs(spectrum) * 2 / SIZE # sum(window)
        spectrum[spectrum == 0] = 1E-6
        spectrum = 10 * np.log(spectrum / 2**15)

        return spectrum


    def updateData(self):

        print(len(self.deque))

        if len(self.deque) == 0:
            print('no data in queue')
            return

        n = 0
        newData = []
        while len(self.deque) > 0:
            newData.extend(self.deque.popleft())
            n += self.CHUNK_SIZE // self.FFT_RATE

        print(self.CHUNK_SIZE // self.FFT_RATE)

        self.audioData = np.append(self.audioData, newData)
        self.audioData = self.audioData[-self.TIME_AXIS_SAMPLES:]

        # process n spectra for every GUI update
        #n = self.CHUNK_SIZE // self.FFT_RATE
        for i in range(n):

            x = self.FFT_SIZE + (n - 1 - i) * self.FFT_RATE
            y = (n - 1 - i) * self.FFT_RATE

            fftData = self.audioData[-x:] if y == 0 else self.audioData[-x:-y]

            self.spectrum = self.getSpectrum(fftData)
            self.maxSpectrum = np.maximum(self.maxSpectrum, self.spectrum)

            create log interpolation of the spectrum
            SIZE = self.FFT_SIZE // 2 + 1
            linRange = np.array(np.arange(SIZE))
            logRange = (1 - np.log10(SIZE - linRange) / np.log10(SIZE)) * (SIZE - 1)

            f = interp1d(linRange, self.spectrum, kind='cubic')

            # print('')
            # print(linRange)
            # print(logRange)
            # print(f(linRange))
            # print(f(logRange))

            self.spectrogram = self.spectrogram[:,1:]
            self.spectrogram = np.column_stack([self.spectrogram, f(logRange)])
            #self.spectrogram = np.column_stack([self.spectrogram, self.spectrum])

        # create higher precision spectrum for score
        self.longSpectrum = self.getSpectrum(self.audioData[-self.LONG_FFT_SIZE:])
        maxPower = float(np.amax(self.longSpectrum))
        if maxPower > self.maxPower:
            self.maxPower = maxPower
            self.maxFrequency = float(np.where(
                self.longSpectrum == maxPower)[0][0] * self.LONG_FFT_RESOLUTION)

        self.updatePlots()


    def updatePlots(self):

        self.plots.lblPower.setText('{:.2f} dBFS'.format(self.maxPower))
        self.plots.lblFrequency.setText('{:.2f} Hz'.format(self.maxFrequency))

        self.spectrogramImage.setImage(self.spectrogram, autoLevels=False)
        self.timeCurve.setData(self.timeAxis, self.audioData[::self.DECIMATION])
        self.spectrumCurve.setData(self.spectrum, self.spectrumScale)
        self.spectrumMaxCurve.setData(self.maxSpectrum, self.spectrumScale, 
            pen=pg.mkPen('r'))


    def clearLayout(self, layout):

        for i in reversed(range(layout.count())): 

            item = layout.itemAt(i)

            if isinstance(item, QHBoxLayout):
                self.clearLayout(item)
            elif isinstance(item, QSpacerItem):
                layout.removeItem(item)
            else:
                widget = item.widget()
                layout.removeWidget(widget)
                widget.setParent(None)


    def updateRanking(self, layout, scores, index, unit, reverse=False):

        self.clearLayout(layout)
        ranking = sorted(scores, key=lambda x: scores[x][index], reverse=reverse)
        font = QFont("Sans Serif", 12) 

        for i, name in enumerate(ranking[:10]):
            
            nameLabel = QLabel('{}'.format(name))
            nameLabel.setAlignment(Qt.AlignLeft)
            nameLabel.setFont(font)

            scoreLabel = QLabel('{:.2f} {}'.format(scores[name][index], unit))
            scoreLabel.setAlignment(Qt.AlignRight)
            scoreLabel.setFont(font)

            entry = QHBoxLayout()
            entry.addWidget(nameLabel)
            entry.addWidget(scoreLabel)
            layout.addLayout(entry)

        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))


    def updateScores(self):

        self.updateRanking(self.scores.boxGlobalLowest.layout(), self.globalScores, 1, 'Hz')
        self.updateRanking(self.scores.boxGlobalLoudest.layout(), self.globalScores, 0, 'dBFS', 
            reverse=True)
        self.updateRanking(self.scores.boxGlobalHighest.layout(), self.globalScores, 1, 'Hz', 
            reverse=True)
        
        self.updateRanking(self.scores.boxLocalLowest.layout(), self.localScores, 1, 'Hz')
        self.updateRanking(self.scores.boxLocalLoudest.layout(), self.localScores, 0, 'dBFS', 
            reverse=True)
        self.updateRanking(self.scores.boxLocalHighest.layout(), self.localScores, 1, 'Hz', 
            reverse=True)
        
        

    def streamCallback(self, inputData, frameCount, timeInfo, status):

        data = np.fromstring(inputData, np.int16)
        self.deque.append(data)

        return (inputData, pyaudio.paContinue)


    def getStream(self):

        pa = pyaudio.PyAudio()
        hostInfo = pa.get_host_api_info_by_index(0)
        nDevices = hostInfo['deviceCount']
        devices = \
            [(x, pa.get_device_info_by_host_api_device_index(0, x)) for x in range(nDevices)]
        #self.pp.pprint(devices)
        inputDevices = list(filter(lambda x: x[1]['maxInputChannels'] > 0, devices))
        usbMicrophones = \
            list(filter(lambda x: 'default' in x[1]['name'], inputDevices))
            #list(filter(lambda x: 'USB PnP Audio Device' in x[1]['name'], inputDevices))

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
        self.timer.start()


    def stop(self):
        self.timer.stop()


    def save(self):
        self.timer.stop()
        self.deque = deque()

        name = self.plots.lineName.text()

        # update highscores
        self.globalScores[name] = [self.maxPower, self.maxFrequency]
        with open('globalscores.yaml', 'w') as f:
            yaml.dump(self.globalScores, f)

        self.localScores[name] = [self.maxPower, self.maxFrequency]
        with open('localscores.yaml', 'w') as f:
            yaml.dump(self.localScores, f)

        # update scores window
        self.updateScoresSignal.emit()

        # create rotated version of spectrum
        spectrumPlot = pg.PlotWidget()
        spectrumPlot.setTitle(title='Power Spectral Density', **self.titleStyle)
        spectrumPlot.setLabel('bottom', 'frequency (Hz)', **self.labelStyle)
        spectrumPlot.setLabel('left', 'power spectral density (dBFS)', **self.labelStyle)
        spectrumPlot.getAxis('left').tickFont = self.font
        spectrumPlot.getAxis('bottom').tickFont = self.font
        spectrumPlot.getPlotItem().setLogMode(True, False)
        spectrumPlot.getPlotItem().setRange(
            xRange=[0, np.floor(np.log10(self.SAMPLE_RATE // 2))], 
            yRange=[self.SPECTRUM_MIN, self.SPECTRUM_MAX])
        spectrumPlot.getAxis('left').setTicks(
            [[(x, str(x)) for x in range(self.SPECTRUM_MIN, self.SPECTRUM_MAX, 20)]])
        spectrumCurve = spectrumPlot.plot()
        spectrumCurve.setData(self.spectrumScale, self.maxSpectrum)

        # export plots to png
        ImageExporter(spectrumPlot.plotItem).export('spectrum.png')
        ImageExporter(self.plots.timePlot.plotItem).export('timeseries.png')
        ImageExporter(self.plots.spectrogramPlot.plotItem).export('spectrogram.png')


    def reset(self):
        self.initPlotData()
        self.deque.append(np.zeros(self.CHUNK_SIZE))
        self.updatePlotsSignal.emit()


def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    qApp.exec_()
    

if __name__ == '__main__':
    main()
