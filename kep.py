#!/usr/bin/env python3

import os
import sys
import time
import pprint
import pyaudio
import yaml
import time
from subprocess import Popen, run, check_output
from datetime import datetime
from collections import deque

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import interp1d
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QSpacerItem
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, Qt
from PyQt5.QtGui import QFont, QSizePolicy, QFont, QPixmap, QIntValidator
from PyQt5.uic import loadUi

# TODO:
# - plot ticks
# - grid
# - minor ticks
# - pdf
# - same name popup
# - save wav


class KrijsEenPrijs(QObject):

    TIME_AXIS_LENGTH        = 10   # s
    SAMPLE_RATE             = 48000 # Hz
    DECIMATION              = 32    # sample decimation factor for time plot
    CHUNK_SIZE              = 2**11 # determines max plot framerate -> ~23.4 Hz
    FFT_RATE                = 2**10 # length of new audio data for each spectral update -> ~46.9 Hz time resolution in spectrogram
    FFT_SIZE                = 2**12 # old samples are used to pad fft 
    LONG_FFT_SIZE           = 2**16 # high resolution fft used only for scores
    FFT_RESOLUTION          = SAMPLE_RATE / FFT_SIZE
    LONG_FFT_RESOLUTION     = SAMPLE_RATE / LONG_FFT_SIZE
    SPECTRUM_MIN            = -20
    SPECTRUM_MAX            = 120
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    UPDATE_FREQ             = SAMPLE_RATE // CHUNK_SIZE
    CALIBRATION             = 120
    DEFAULT_PRINTER         = 'PU04'

    updateDataSignal = pyqtSignal()
    updateScoresSignal = pyqtSignal()

    def __init__(self):

        super(KrijsEenPrijs, self).__init__()

        self.pp = pprint.PrettyPrinter()
        self.deque = deque()
        self.running = True
        self.t0 = datetime.now()
        self.maxFrequency = 0
        self.maxPower = self.SPECTRUM_MIN
        self.spectrumScale = np.linspace(0, self.SAMPLE_RATE / 2, self.FFT_SIZE / 2)
        self.timeAxis = np.linspace(
            -self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES / self.DECIMATION)
        self.logScaleSpectrogram = (np.log(range(1, self.FFT_SIZE // 2 + 1)) 
            / np.log(self.FFT_SIZE // 2 + 1) * self.FFT_SIZE // 2)

        try:
            with open('scores/globalscores.yaml') as f:
                self.globalScores = yaml.load(f)
            if self.globalScores == None:
                self.globalScores = {}
        except:
            print('could not load global highscores')
            self.globalScores = {}

        try:
            with open('scores/localscores.yaml') as f:
                self.localScores = yaml.load(f)
            if self.localScores == None:
                self.localScores = {}
        except:
            print('could not load local highscores')
            self.localScores = {}
        
        self.initPlotData()
        self.initPlotsGui()
        self.initScoresGui()
        self.updateDataSignal.connect(self.updateData)
        self.updateScoresSignal.connect(self.updateScores)
        self.stream = self.getStream()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateData)
        self.updateScoresSignal.emit()
        #time.sleep(1)
        self.timer.start(self.CHUNK_SIZE / self.SAMPLE_RATE * 1000 * 0.95)


    def initPlotData(self):

        self.maxPower    = self.SPECTRUM_MIN
        self.audioData   = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.spectrum    = np.ones(self.FFT_SIZE // 2, dtype=np.int16) * self.SPECTRUM_MIN
        self.maxSpectrum = np.ones(self.FFT_SIZE // 2, dtype=np.int16) * self.SPECTRUM_MIN
        self.spectrogram = np.ones((self.FFT_SIZE // 2, 
                            self.TIME_AXIS_SAMPLES // self.FFT_RATE),
                            dtype=np.float32) * self.SPECTRUM_MIN


    def initPlotsGui(self):
        
        pg.setConfigOptions(
            imageAxisOrder='row-major', background=pg.mkColor(0x0, 0x0, 0x100, 0x24))

        self.plots = loadUi('ui/plots.ui')

        self.font = QFont()
        self.font.setPixelSize(14)
        self.labelStyle = {'color': '#FFF', 'font-size': '16px'}
        self.titleStyle = {'color': '#FFF', 'font-size': '40px'}

        self.plots.labelLogo.setPixmap(QPixmap('images/sron_small.png'))

        self.plots.timePlot.setTitle('Microphone Signal', **self.titleStyle)
        self.plots.timePlot.setLabel('left', 'amplitude', **self.labelStyle)
        self.plots.timePlot.setLabel('bottom', 'time (s)', **self.labelStyle)
        self.plots.timePlot.getAxis('left').tickFont = self.font
        self.plots.timePlot.getAxis('bottom').tickFont = self.font
        self.plots.timePlot.setRange(yRange=[1.2 * -2**15, 1.2 * 2**15])
        self.plots.timePlot.setMouseEnabled(x=False, y=False)
        
        self.plots.spectrumPlot.setTitle(title='Power Spectral Density', **self.titleStyle)
        self.plots.spectrumPlot.setLabel('left', 'frequency (Hz)', **self.labelStyle)
        self.plots.spectrumPlot.setLabel(
            'bottom', 'power spectral density (dB', **self.labelStyle)
        self.plots.spectrumPlot.getAxis('left').tickFont = self.font
        self.plots.spectrumPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrumPlot.getPlotItem().setLogMode(False, True)
        self.plots.spectrumPlot.getPlotItem().setRange(
            xRange=[self.SPECTRUM_MIN, self.SPECTRUM_MAX])

        self.plots.spectrumPlot.getAxis('bottom').setTicks(
            [[(x, str(x)) for x in range(self.SPECTRUM_MIN, self.SPECTRUM_MAX + 20, 20)]])

        self.plots.spectrumPlot.getAxis('left').setTicks(
            [[(x, str(int(10**x))) for x in [1, 2, 3, 4, 5]]])

        self.plots.spectrumPlot.setMouseEnabled(x=False, y=False)
        
        self.timeCurve = self.plots.timePlot.plot()
        self.spectrumCurve = self.plots.spectrumPlot.plot()
        self.spectrumMaxCurve = self.plots.spectrumPlot.plot()
        
        self.spectrogramImage = pg.ImageItem()
        self.spectrogramImage.setLevels([self.SPECTRUM_MIN, self.SPECTRUM_MAX])
        self.plots.spectrogramPlot.setTitle('Spectrogram', **self.titleStyle)
        # self.plots.spectrogramPlot.setLabel('left', 'FFT bin', **self.labelStyle)
        # self.plots.spectrogramPlot.setLabel('bottom', 'time (s)', **self.labelStyle)
        self.plots.spectrogramPlot.setLabel('left', ' ', **self.labelStyle)
        self.plots.spectrogramPlot.setLabel('bottom', ' ', **self.labelStyle)
        self.plots.spectrogramPlot.getAxis('left').tickFont = self.font
        self.plots.spectrogramPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrogramPlot.setMouseEnabled(x=False, y=False)
        self.plots.spectrogramPlot.addItem(self.spectrogramImage)

        pos = np.array([0., 0.3, 0.7, 1.0])
        color = np.array([[0x0,  0x0,  0x24, 0xff], 
                          [0x17, 0x29, 0xc3, 0xff], 
                          [0xd0, 0x24, 0x12, 0xff], 
                          [0xff, 0xc4, 0x0,  0xff]], dtype=np.ubyte)

        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spectrogramImage.setLookupTable(lut)

        ageValidator = QIntValidator(0, 1000)
        self.plots.lineAge.setValidator(ageValidator)

        # populate printer dropdown menu 
        outp = check_output(['lpstat', '-p', '-d'])
        print(outp)
        print(outp.split(b'\n'))
        for line in outp.split(b'\n'):
            print(line)
            words = line.split(b' ')
            if words[0] == b'printer':
                printername = words[1].strip().decode('utf-8')
                self.plots.boxPrinter.addItem(printername)

        index = self.plots.boxPrinter.findText(self.DEFAULT_PRINTER)
        if index:
            self.plots.boxPrinter.setCurrentIndex(index)

        self.plots.startButton.clicked.connect(self.start)
        self.plots.stopButton.clicked.connect(self.stop)
        self.plots.saveButton.clicked.connect(self.save)
        self.plots.resetButton.clicked.connect(self.reset)

        self.plots.setWindowTitle('Plots')
        self.plots.resize(1920, 1200)
        self.plots.show()


    def initScoresGui(self):

        self.scores = loadUi('ui/scores.ui')

        self.scores.powerHistogram.setTitle('Power Score Distribution', **self.titleStyle)
        self.scores.powerHistogram.setLabel('left', '# of people', **self.labelStyle)
        self.scores.powerHistogram.setLabel('bottom', 'Power (dB)', **self.labelStyle)
        self.scores.powerHistogram.getAxis('left').tickFont = self.font
        self.scores.powerHistogram.getAxis('bottom').tickFont = self.font
        self.scores.powerHistogram.setMouseEnabled(x=False, y=False)

        self.scores.frequencyHistogram.setTitle('Main Frequency Distribution', **self.titleStyle)
        self.scores.frequencyHistogram.setLabel('left', '# of people', **self.labelStyle)
        self.scores.frequencyHistogram.setLabel('bottom', 'frequency (Hz)', **self.labelStyle)
        self.scores.frequencyHistogram.getAxis('left').tickFont = self.font
        self.scores.frequencyHistogram.getAxis('bottom').tickFont = self.font
        self.scores.frequencyHistogram.setMouseEnabled(x=False, y=False)

        self.scores.boxPowerBins.valueChanged.connect(self.updateScores)
        self.scores.boxFrequencyBins.valueChanged.connect(self.updateScores)
        self.scores.actionNewRound.triggered.connect(self.clearRound)

        self.scores.setWindowTitle('High Scores')
        self.scores.resize(1920, 1200)
        self.scores.show()


    def getSpectrum(self, data):

        SIZE = len(data)

        data = data * np.hanning(SIZE) 

        spectrum = np.fft.rfft(data[-SIZE:])
        spectrum = np.abs(spectrum) * 2 / SIZE # sum(window)
        #spectrum = spectrum**2
        spectrum[spectrum == 0] = 1E-6
        spectrum = 20 * np.log10(spectrum / 2**15)
        spectrum += self.CALIBRATION
        spectrum = np.clip(spectrum, self.SPECTRUM_MIN, self.SPECTRUM_MAX)

        return spectrum[1:] # drop DC bin


    def updateData(self):

        #print('l = {}'.format(len(self.deque)))

        if len(self.deque) == 0:
            return

        elif len(self.deque) < 3:
            n = self.CHUNK_SIZE // self.FFT_RATE
            newData = self.deque.popleft()

        else: # catch up plotting
            n = 0
            newData = np.array([])

            while len(self.deque) > 1:
                n += self.CHUNK_SIZE // self.FFT_RATE
                newData = np.concatenate((newData, self.deque.popleft()))


        self.audioData = np.roll(self.audioData, -len(newData))
        self.audioData[-len(newData):] = newData

        # process n spectra for every GUI update
        for i in range(n):

            x = self.FFT_SIZE + (n - 1 - i) * self.FFT_RATE
            y = (n - 1 - i) * self.FFT_RATE

            fftData = self.audioData[-x:] if y == 0 else self.audioData[-x:-y]

            self.spectrum = self.getSpectrum(fftData)
            self.maxSpectrum = np.maximum(self.maxSpectrum, self.spectrum)

            # create log interpolation of the spectrum
            SIZE = self.FFT_SIZE // 2

            linRange = np.array(np.arange(SIZE))
            logRange = 10**(np.log10(SIZE) / SIZE * linRange) # (magic)
            f = interp1d(linRange, self.spectrum, kind='cubic')

            self.spectrogram = self.spectrogram[:,1:]
            self.spectrogram = np.column_stack([self.spectrogram, f(logRange)])

        # create higher precision spectrum for score
        self.longSpectrum = self.getSpectrum(self.audioData[-self.LONG_FFT_SIZE:])
        maxPower = np.amax(self.longSpectrum)
        if maxPower > self.maxPower:
            self.maxPower = float(maxPower)
            self.maxFrequency = float(np.where(
                self.longSpectrum == maxPower)[0][0]) * self.LONG_FFT_RESOLUTION

        self.updatePlots()


    def updatePlots(self):

        self.plots.lblPower.setText('{:.2f} dB'.format(self.maxPower))
        self.plots.lblFrequency.setText('{:.2f} Hz'.format(self.maxFrequency))

        self.spectrogramImage.setImage(self.spectrogram, autoLevels=False)
        self.timeCurve.setData(self.timeAxis, self.audioData[::self.DECIMATION])
        self.spectrumCurve.setData(self.spectrum, self.spectrumScale)
        self.spectrumMaxCurve.setData(self.maxSpectrum, self.spectrumScale, 
            pen=pg.mkPen('r'))


    def clearRound(self):

        print('clear')

        self.localScores = {} 
        self.dumpScores()
        self.updateScores()


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

        print('updatescores')
        print(self.localScores)

        self.updateRanking(self.scores.boxGlobalLowest.layout(), self.globalScores, 2, 'Hz')
        self.updateRanking(self.scores.boxGlobalHighest.layout(), self.globalScores, 2, 'Hz', 
            reverse=True)
        self.updateRanking(self.scores.boxGlobalLoudest.layout(), self.globalScores, 1, 'dB', 
            reverse=True)
        
        self.updateRanking(self.scores.boxLocalLowest.layout(), self.localScores, 2, 'Hz')
        self.updateRanking(self.scores.boxLocalHighest.layout(), self.localScores, 2, 'Hz', 
            reverse=True)
        self.updateRanking(self.scores.boxLocalLoudest.layout(), self.localScores, 1, 'dB', 
            reverse=True)


        powerScores = [x[1] for x in self.globalScores.values()]
        y, x = np.histogram(powerScores, bins=np.linspace(
            np.min(powerScores), np.max(powerScores), self.scores.boxPowerBins.value()))
        self.scores.powerHistogram.clear()
        self.scores.powerHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

        frequencyScores = [x[2] for x in self.globalScores.values()]
        y, x = np.histogram(frequencyScores, bins=np.linspace(
                np.min(frequencyScores), np.max(frequencyScores), self.scores.boxFrequencyBins.value()))
        self.scores.frequencyHistogram.clear()
        self.scores.frequencyHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        
        

    def streamCallback(self, inputData, frameCount, timeInfo, status):

        if self.timer.isActive():
            data = np.fromstring(inputData, np.int16)
            self.deque.append(data)
        # print('emit')
        # self.updateDataSignal.emit()

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
            list(filter(lambda x: 'USB PnP Audio Device' in x[1]['name'], inputDevices))
            #list(filter(lambda x: 'default' in x[1]['name'], inputDevices))

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
        self.timer.start(self.CHUNK_SIZE / self.SAMPLE_RATE * 1000)


    def stop(self):
        self.timer.stop()


    def save(self):
        self.timer.stop()
        self.deque = deque()

        firstName = self.plots.lineName.text()
        firstName = firstName[0].upper() + firstName[1:]

        lastName = self.plots.lineSurname.text().split(' ')
        lastName[-1] = lastName[-1][0].upper() + lastName[-1][1:]
        lastName = ' '.join(lastName)

        name = '{} {}'.format(firstName, lastName)

        self.globalScores[name] = [int(self.plots.lineAge.text()), self.maxPower, self.maxFrequency]
        self.localScores[name] = [int(self.plots.lineAge.text()), self.maxPower, self.maxFrequency]

        # update highscores
        self.dumpScores()

        # update scores window
        self.updateScoresSignal.emit()

        pen = pg.mkPen(color=(0, 0, 0), width=3)
        pg.setConfigOption('background', 'w')

        timePlot = pg.PlotWidget()
        timePlot.setRange(yRange=[1.2 * -2**15, 1.2 * 2**15])
        timePlot.setBackground(None)
        timePlot.hideAxis('left')
        timePlot.hideAxis('bottom')
        timePlot.resize(800,200)
        timeCurve = timePlot.plot(self.timeAxis, self.audioData[::self.DECIMATION], pen=pen)

        # create rotated version of spectrum
        spectrumPlot = pg.PlotWidget()
        spectrumPlot.setBackground(None)
        spectrumPlot.getPlotItem().setLogMode(True, False)
        spectrumPlot.getPlotItem().setRange(yRange=[self.SPECTRUM_MIN, self.SPECTRUM_MAX])
        spectrumPlot.getAxis('left').setTicks(
            [[(x, str(x)) for x in range(self.SPECTRUM_MIN, self.SPECTRUM_MAX, 20)]])
        spectrumPlot.getAxis('bottom').setTicks([[(x, str(int(10**x))) for x in [1, 2, 3, 4, 5]]])
        spectrumCurve = spectrumPlot.plot(self.spectrumScale, self.maxSpectrum)

        export plots to png
        exporter = ImageExporter(spectrumPlot.plotItem)
        exporter.parameters()['width'] = 2000
        exporter.export('images/spectrum.png')

        exporter = ImageExporter(timePlot.plotItem)
        exporter.parameters()['width'] = 2000
        exporter.parameters()['width'] = 2000
        exporter.export('images/timeseries.png')
        
        exporter = ImageExporter(self.spectrogramImage)
        exporter.parameters()['width'] = 2000
        exporter.export('images/spectrogram.png')

        time.sleep(0.5)
        run(['lp', '-d', self.plots.boxPrinter.currentText(), './spectrogram.png'])


    def dumpScores(self):
        
        #os.remove('globalscores.yaml')
        with open('scores/globalscores.yaml', 'w') as f:
            yaml.dump(self.globalScores, f)

        #os.remove('localscores.yaml')
        with open('scores/localscores.yaml', 'w') as f:
            yaml.dump(self.localScores, f)


    def reset(self):
        self.initPlotData()
        self.deque.append(np.zeros(self.CHUNK_SIZE))
        self.updateDataSignal.emit()


def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()
    qApp.exec_()
    

if __name__ == '__main__':
    
    main()
