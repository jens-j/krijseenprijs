#!/usr/bin/env python3

import os
import sys
import time
import pprint
import pyaudio
import yaml
import time
from subprocess import Popen, run, check_output, CalledProcessError
from datetime import datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import interp1d
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, Qt, QEvent
from PyQt5.QtGui import QFont, QFont, QPixmap, QIntValidator
from PyQt5.uic import loadUi

from pdf import createDiploma

# TODO:
# - better dB scaling
# - throw away highest bins?
# - start / stop using spacebar -> wont work


class MyQWidget(QWidget):

    spaceBarClicked = pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self)


    # Toggle start/stop with the spacebar.
    def keyPressEvent(self, event):

        print('keyPressEvent')
        if event.key() == Qt.Key_Space:
            event.accept()
            self.spaceBarClicked.emit()
        else:
            event.ignore()



class KrijsEenPrijs(QMainWindow):

    TIME_AXIS_LENGTH        = 8    # s
    SAMPLE_RATE             = 48000 # Hz
    DECIMATION              = 32    # sample decimation factor for time plot
    CHUNK_SIZE              = 2**11 # determines max plot framerate -> ~23.4 Hz
    FFT_RATE                = 2**10 # length of new audio data for each spectral update -> ~46.9 Hz time resolution in spectrogram
    FFT_SIZE                = 2**12 # old samples are used to pad fft
    LONG_FFT_SIZE           = 2**16 # high resolution fft used only for scores
    FFT_RESOLUTION          = SAMPLE_RATE / FFT_SIZE
    LONG_FFT_RESOLUTION     = SAMPLE_RATE / LONG_FFT_SIZE
    SPECTRUM_MIN_COLORMAP   = 0
    SPECTRUM_MIN_GRAPH      = -20
    SPECTRUM_MAX            = 100
    TIME_AXIS_SAMPLES       = SAMPLE_RATE * TIME_AXIS_LENGTH
    UPDATE_FREQ             = SAMPLE_RATE // CHUNK_SIZE
    SPECTROGRAM_WIDTH       = TIME_AXIS_SAMPLES // FFT_RATE
    SPECTROGRAM_HEIGHT      = FFT_SIZE // 2
    CALIBRATION             = 120
    DEFAULT_PRINTER         = 'PL04'
    NOTES                   = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    updateDataSignal = pyqtSignal()
    updateScoresSignal = pyqtSignal()


    def __init__(self, parent=None):

        QMainWindow.__init__(self, parent)

        self.uipath = os.path.dirname(os.path.abspath(__file__)) + '/ui/'
        self.pp = pprint.PrettyPrinter()
        self.deque = deque()
        self.running = True
        self.t0 = datetime.now()
        self.maxFrequency = 0
        self.totalPower = 0
        self.note = '--'
        self.spectrumScale = np.linspace(0, self.SAMPLE_RATE // 2, self.FFT_SIZE // 2)

        self.timeAxis = np.linspace(
            -self.TIME_AXIS_LENGTH, 0, self.TIME_AXIS_SAMPLES // self.DECIMATION)

        self.logScaleSpectrogram = (np.log(range(1, self.FFT_SIZE // 2 + 1))
            / np.log(self.FFT_SIZE // 2 + 1) * self.FFT_SIZE // 2)

        self.spectrogramLinRange = np.array(np.arange(self.FFT_SIZE // 2))
        self.spectrogramLogRange = 10**(np.log10(self.FFT_SIZE // 2) / (self.FFT_SIZE // 2) * self.spectrogramLinRange) # (magic)

        print(self.spectrogramLinRange)
        print(self.spectrogramLogRange)
        print(len(self.spectrogramLogRange))

        try:
            with open('scores/globalscores.yaml') as f:
                self.globalScores = yaml.load(f, Loader=yaml.CLoader)
            if self.globalScores == None:
                self.globalScores = {}
        except Exception as e:
            print(f'could not load global highscores ({e})')
            self.globalScores = {}

        try:
            with open('scores/localscores.yaml') as f:
                self.localScores = yaml.load(f, Loader=yaml.CLoader)
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

        # populate score screen
        if not (self.globalScores == {} and self.localScores == {}):
            self.updateScoresSignal.emit()

        self.timer.start(int(self.CHUNK_SIZE / self.SAMPLE_RATE * 1000 * 0.95))


    def initPlotData(self):

        self.totalPower  = self.SPECTRUM_MIN_COLORMAP
        self.audioData   = np.zeros(self.TIME_AXIS_SAMPLES, dtype=np.int16)
        self.spectrum    = np.ones(self.FFT_SIZE // 2, dtype=np.int16) * self.SPECTRUM_MIN_COLORMAP
        self.maxSpectrum = np.ones(self.FFT_SIZE // 2, dtype=np.int16) * self.SPECTRUM_MIN_COLORMAP
        self.spectrogram = np.ones((self.SPECTROGRAM_HEIGHT, self.SPECTROGRAM_WIDTH), dtype=np.float32) \
            * self.SPECTRUM_MIN_COLORMAP


    def initPlotsGui(self):

        pg.setConfigOptions(
            imageAxisOrder='row-major', background=pg.mkColor(0x0, 0x0, 0x100, 0x24))

        self.plots = loadUi(f'{self.uipath}/plots_widget_grid.ui')

        self.font = QFont()
        self.font.setPixelSize(14)
        self.labelStyle = {'color': '#FFF', 'font-size': '16px'}
        self.titleStyle = {'color': '#FFF', 'font-size': '40px'}

        self.plots.labelLogo.setPixmap(QPixmap('images/sron_small_bg.png'))

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
            'bottom', 'power spectral density (dB)', **self.labelStyle)
        self.plots.spectrumPlot.getAxis('left').tickFont = self.font
        self.plots.spectrumPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrumPlot.getPlotItem().setLogMode(False, True)
        self.plots.spectrumPlot.getPlotItem().setRange(
            xRange=[self.SPECTRUM_MIN_GRAPH, self.SPECTRUM_MAX])

        self.plots.spectrumPlot.getAxis('bottom').setTicks(
            [[(x, str(x)) for x in range(self.SPECTRUM_MIN_COLORMAP, self.SPECTRUM_MAX + 20, 20)]])

        self.plots.spectrumPlot.getAxis('left').setTicks(
            [[(x, str(int(10**x))) for x in [1, 2, 3, 4, 5]]])

        self.plots.spectrumPlot.setMouseEnabled(x=False, y=False)

        self.timeCurve = self.plots.timePlot.plot()
        self.spectrumCurve = self.plots.spectrumPlot.plot()
        self.spectrumMaxCurve = self.plots.spectrumPlot.plot()

        self.spectrogramImage = pg.ImageItem()
        self.spectrogramImage.setLevels([self.SPECTRUM_MIN_COLORMAP, self.SPECTRUM_MAX])
        self.plots.spectrogramPlot.setTitle('Spectrogram', **self.titleStyle)
        self.plots.spectrogramPlot.setLabel('left', 'FFT bin', **self.labelStyle)
        self.plots.spectrogramPlot.setLabel('bottom', 'time (s)', **self.labelStyle)
        # self.plots.spectrogramPlot.setLabel('left', ' ', **self.labelStyle)
        # self.plots.spectrogramPlot.setLabel('bottom', ' ', **self.labelStyle)
        self.plots.spectrogramPlot.getAxis('left').tickFont = self.font
        self.plots.spectrogramPlot.getAxis('bottom').tickFont = self.font
        self.plots.spectrogramPlot.setMouseEnabled(x=False, y=False)
        self.plots.spectrogramPlot.addItem(self.spectrogramImage)

        # Convert pixel indices to seconds for spectrogram ticks.
        xTickTo = [str(x) for x in range(-8, 1)]
        xTickFrom = [x * self.SPECTROGRAM_WIDTH / 8 for x in range(9)]
        xTicks = dict(zip(xTickFrom, xTickTo))
        xAxis = self.plots.spectrogramPlot.getAxis('bottom')
        xAxis.setTicks([xTicks.items()])

        # Convert pixel indices to frequency for spectrogram ticks.
        yTickTo = [10, 100, 1000, 10000]
        axisInterp = interp1d(self.spectrogramLogRange, self.spectrogramLinRange, kind='cubic')
        yTickFrom = [int(axisInterp(max(1, x * self.SPECTROGRAM_HEIGHT / (self.SAMPLE_RATE / 2)))) for x in yTickTo]
        yTicks = dict(zip(yTickFrom, [str(x) for x in yTickTo]))
        yAxis = self.plots.spectrogramPlot.getAxis('left')
        yAxis.setTicks([yTicks.items()])


        pos = np.array([0., 0.3, 0.7, 1.0])
        color = np.array([[0x0,  0x0,  0x24, 0xff],
                          [0x17, 0x29, 0xc3, 0xff],
                          [0xd0, 0x24, 0x12, 0xff],
                          [0xff, 0xc4, 0x0,  0xff]], dtype=np.ubyte)


        cmap = pg.colormap.get('viridis')
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spectrogramImage.setLookupTable(lut)

        ageValidator = QIntValidator(0, 1000)
        self.plots.lineAge.setValidator(ageValidator)

        # populate printer dropdown menu
        try:
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
        except CalledProcessError as e:
            print(f'lpstat error: {e}')

        self.plots.startButton.clicked.connect(self.start)
        self.plots.stopButton.clicked.connect(self.stop)
        self.plots.saveButton.clicked.connect(self.save)
        self.plots.resetButton.clicked.connect(self.reset)
        self.plots.keypress_widget.spaceBarClicked.connect(self.handleSpaceBarClicked)

        self.plots.setWindowTitle('Plots')
        self.plots.resize(1920, 1080)

        self.plots.show()


    def initScoresGui(self):

        self.scores = loadUi(f'{self.uipath}/scores.ui')

        self.scores.powerHistogram.setTitle('Power Score Distribution', **self.titleStyle)
        self.scores.powerHistogram.setLabel('left', '# of subjects', **self.labelStyle)
        self.scores.powerHistogram.setLabel('bottom', 'Power (dB)', **self.labelStyle)
        self.scores.powerHistogram.getAxis('left').tickFont = self.font
        self.scores.powerHistogram.getAxis('bottom').tickFont = self.font
        self.scores.powerHistogram.setMouseEnabled(x=False, y=False)

        self.scores.frequencyHistogram.setTitle('Main Frequency Distribution', **self.titleStyle)
        self.scores.frequencyHistogram.setLabel('left', '# of subjects', **self.labelStyle)
        self.scores.frequencyHistogram.setLabel('bottom', 'frequency (Hz)', **self.labelStyle)
        self.scores.frequencyHistogram.getAxis('left').tickFont = self.font
        self.scores.frequencyHistogram.getAxis('bottom').tickFont = self.font
        self.scores.frequencyHistogram.setMouseEnabled(x=False, y=False)

        self.scores.boxPowerBins.valueChanged.connect(self.updateScores)
        self.scores.boxFrequencyBins.valueChanged.connect(self.updateScores)
        self.scores.actionNewRound.triggered.connect(self.clearRound)

        self.scores.setWindowTitle('High Scores')
        self.scores.resize(1920, 1080)
        self.scores.show()


    def getSpectrum(self, data):

        SIZE = len(data)

        data = data * np.hanning(SIZE)

        spectrum = np.fft.rfft(data[-SIZE:])
        spectrum = np.abs(spectrum) * 2 / SIZE
        spectrum[spectrum == 0] = 1E-6
        spectrum = 20 * np.log10(spectrum / 2**15)
        spectrum += self.CALIBRATION
        spectrum = np.clip(spectrum, self.SPECTRUM_MIN_GRAPH, self.SPECTRUM_MAX)

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
            self.spectrogramInterp = interp1d(self.spectrogramLinRange, self.spectrum, kind='cubic')
            self.spectrogram = self.spectrogram[:,1:]
            self.spectrogram = np.column_stack([self.spectrogram, self.spectrogramInterp(self.spectrogramLogRange)])

        # create higher precision spectrum for score
        self.longSpectrum = self.getSpectrum(self.audioData[-self.LONG_FFT_SIZE:])
        maxPower = np.amax(self.longSpectrum)
        totalPower = np.average(self.longSpectrum)
        if totalPower > self.totalPower:
            self.totalPower = float(totalPower)
            self.maxFrequency = float(np.where(
                self.longSpectrum == maxPower)[0][0]) * self.LONG_FFT_RESOLUTION
            self.note = self.getNote(self.maxFrequency)

        self.updatePlots()


    def updatePlots(self):

        self.plots.lblPower.setText(f'{self.totalPower:.2f} dB')
        self.plots.lblFrequency.setText(f'{self.maxFrequency:.2f} Hz')
        self.plots.lblNote.setText(f'{self.note}')

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
            np.min(powerScores), np.max(powerScores), self.scores.boxPowerBins.value() + 1))
        self.scores.powerHistogram.clear()
        self.scores.powerHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

        frequencyScores = [x[2] for x in self.globalScores.values()]
        y, x = np.histogram(frequencyScores, bins=np.linspace(
                np.min(frequencyScores),
                np.max(frequencyScores),
                self.scores.boxFrequencyBins.value() + 1))
        self.scores.frequencyHistogram.clear()
        self.scores.frequencyHistogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))



    def streamCallback(self, inputData, frameCount, timeInfo, status):

        if self.timer.isActive():
            data = np.frombuffer(inputData, np.int16)
            self.deque.append(data)
        # print('emit')
        # self.updateDataSignal.emit()

        return (inputData, pyaudio.paContinue)


    def getStream(self):

        pa = pyaudio.PyAudio()
        devices = []
        device_count = pa.get_device_count()
        print('device_count', device_count)

        for i in range(device_count):
            devices.append(pa.get_device_info_by_index(i))
            print(i, devices[-1]['name'], devices[-1]['maxInputChannels'])

        inputDevices = list(filter(lambda x: x['maxInputChannels'] > 0, devices))
        usbMicrophones = \
            list(filter(lambda x: 'USB PnP Audio Device' in x['name'], inputDevices))

        assert len(usbMicrophones) > 0, 'No microphone detected'
        print(usbMicrophones)
        mic = usbMicrophones[0]

        return pa.open(format=pyaudio.paInt16,
                       input_device_index=mic['index'],
                       channels=1,
                       rate=self.SAMPLE_RATE,
                       frames_per_buffer=self.CHUNK_SIZE,
                       input=True,
                       stream_callback=self.streamCallback)


    def handleSpaceBarClicked(self):

            if self.timer.isActive():
                self.stop()
            else:
                self.start()


    def start(self):
        self.timer.start(self.CHUNK_SIZE // self.SAMPLE_RATE * 1000)


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

        self.globalScores[name] = [int(self.plots.lineAge.text()), self.totalPower, self.maxFrequency]
        self.localScores[name] = [int(self.plots.lineAge.text()), self.totalPower, self.maxFrequency]

        # update highscores
        self.dumpScores()

        # update scores window
        self.updateScoresSignal.emit()

        pen = pg.mkPen(color=(0, 0, 0), width=2)
        pg.setConfigOption('background', 'w')

        timePlot = pg.PlotWidget()
        timePlot.setRange(yRange=[1.2 * -2**15, 1.2 * 2**15])
        timePlot.setBackground(None)
        timePlot.hideAxis('left')
        timePlot.hideAxis('bottom')
        timePlot.resize(800,200)
        timeCurve = timePlot.plot(self.timeAxis, self.audioData[::self.DECIMATION], pen=pen)

        # create rotated version of spectrum
        # labelStyle = {'color': '#000', 'font-size': '16px'}
        # spectrumPlot = pg.PlotWidget()
        # # spectrumPlot.setTitle(title='Power Spectral Density', **labelStyle)
        # # spectrumPlot.setLabel('bottom', 'frequency (Hz)', **labelStyle)
        # # spectrumPlot.setLabel('left', 'power spectral density (dB)', **labelStyle)
        # spectrumPlot.getAxis("left").setWidth(200)
        # spectrumPlot.getAxis('left').setPen(pen)
        # spectrumPlot.getAxis('bottom').setPen(pen)
        # #spectrumPlot.setBackground(None)
        # spectrumPlot.getPlotItem().setLogMode(True, False)
        # spectrumPlot.getPlotItem().setRange(
        #     xRange=[1, np.log10(self.SAMPLE_RATE // 2)],
        #     yRange=[0, self.SPECTRUM_MAX])

        # spectrumPlot.getAxis('left').setTicks(
        #     [[(x, str(x)) for x in range(self.SPECTRUM_MIN_GRAPH, self.SPECTRUM_MAX, 20)]])

        # spectrumPlot.getAxis('bottom').setTicks(
        #     [[(x, str(int(10**x))) for x in [1, 2, 3, 4, 5]]])

        # spectrumCurve = spectrumPlot.plot(self.spectrumScale, self.maxSpectrum, pen=pen)

        # export plots to png
        # exporter = ImageExporter(spectrumPlot.plotItem)
        # exporter.parameters()['width'] = 2000
        # exporter.export('images/spectrum.png')

        exporter = ImageExporter(timePlot.plotItem)
        exporter.parameters()['width'] = 2000
        exporter.export('images/timeseries.png')

        exporter = ImageExporter(self.spectrogramImage)
        exporter.parameters()['width'] = 2000
        exporter.export('images/spectrogram.png')

        diplomaPath = createDiploma(firstName, lastName, self.totalPower, self.maxFrequency, self.note)

        # time.sleep(0.5)
        run(['lp',
             '-o', 'print-quality=5', # use best quality
             '-o', 'fit-to-page',     # avoid clipped edges
             '-d', self.plots.boxPrinter.currentText(), diplomaPath])


    def dumpScores(self):

        with open('scores/globalscores.yaml', 'w') as f:
            yaml.dump(self.globalScores, f)

        with open('scores/localscores.yaml', 'w') as f:
            yaml.dump(self.localScores, f)


    def reset(self):
        self.initPlotData()
        self.deque.append(np.zeros(self.CHUNK_SIZE))
        self.updateDataSignal.emit()


    # Return the closest note for a frequency
    def getNote(self, frequency):

        # calculate the interval in semitones w.r.t C0.
        semitones = int(np.round(np.log2(frequency / 16.35) / (np.log2(2) / 12)))

        octave = semitones // 12
        note = self.NOTES[semitones % 12]

        return f'{note}{octave}'



def main():
    qApp = QApplication(sys.argv)
    kep = KrijsEenPrijs()

    uipath = os.path.dirname(os.path.abspath(__file__)) + '/ui/'

    # Set QSS style
    with open(uipath + 'dark-colorfull.qss') as f:
        styleSheet = f.read();
        qApp.setStyleSheet(styleSheet)

    # kep.show() # The mainwindow is empty so it should be invisible.
    sys.exit(qApp.exec_())


if __name__ == '__main__':

    main()
