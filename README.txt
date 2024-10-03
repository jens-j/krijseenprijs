Python required packages:
matplotlib
pyqtgraph
pyaudio
reportlab

Install using the following command:
apt install python3-pyaudio python3-matplotlib python3-pyqtgraph python3-reportlab


Last time I had issues with detecting the microphone.
Solved using the following commands:
(https://unix.stackexchange.com/questions/263263/remove-pulseaudio-device)

$ pactl list # look for Owner Module: #
$ pactl unload-module #
