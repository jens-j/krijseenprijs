#!/usr/bin/env python3

import locale
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.rl_config import defaultPageSize
from reportlab.pdfbase.pdfmetrics import stringWidth

PAGE_WIDTH  = defaultPageSize[0]


def drawText(c, text, font, points, x, y):

    width = stringWidth(text, font, points)
    textObject = c.beginText()
    textObject.setTextOrigin(x, y)
    textObject.setFont(font, points)
    textObject.textLine(text)
    c.drawText(textObject)


def centerText(c, text, font, points, y):

    width = stringWidth(text, font, points)
    x = (PAGE_WIDTH - width) / 2.0
    drawText(c, text, font, points, x, y)


def createDiploma(name, surname, power, frequency, note):

    locale.setlocale(locale.LC_ALL, 'nl_NL')
    dateString = datetime.strftime(datetime.now(), '%A %-d %B %Y')

    path = 'diplomas/diploma_{}_{}.pdf'.format(name, surname)
    c = canvas.Canvas(path)

    c.drawImage('images/border_new3.png', 0, 0, 21*cm, 29.7*cm)

    centerText(c, 'Krijsdiploma', 'Times-BoldItalic', 60, 23*cm)
    centerText(c, '{} {}'.format(name, surname), 'Times-BoldItalic', 32, 21.3*cm)
    centerText(c, 'Behaald op {}'.format(dateString), 'Times-Italic', 20, 20*cm)
    centerText(c, 'Krijsniveau: {:.2f} dB'.format(power), 'Times-BoldItalic', 24, 19*cm)
    centerText(c, 'Luidste toon: {:.2f} Hz ({})'.format(frequency, note), 'Times-BoldItalic', 24, 18*cm)

    drawText(c, 'Krijs signaal:', 'Times-Italic', 16, 2.5*cm, 16.8*cm)
    drawText(c, 'Krijs spectrogram:', 'Times-Italic', 16, 2.5*cm, 13.2*cm)
    drawText(c, '(10 Hz - 24 kHz, logarithmisch)', 'Times-Italic', 12, 13.5*cm, 4.5*cm)

    c.drawImage('images/timeseries.png', 0.6*cm, 13*cm, 18.8*cm, 4*cm, mask='auto')
    c.drawImage('images/spectrogram.png', 2.3*cm, 5*cm, 16.5*cm, 8*cm)

    c.drawImage('images/logo_logo.png', 15*cm, 1.7*cm, 4*cm, 1.4*cm, mask='auto')
    c.drawImage('images/logo_text.png', 12.4*cm, 1.2*cm, 6.8*cm, 0.35*cm, mask='auto')

    c.showPage()
    c.save()

    return path


if __name__ == '__main__':
    createDiploma('Pietje', 'Puk', 101.0, 800)
