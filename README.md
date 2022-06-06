Live RPI Monitor
----------------

The goal of this project is to make a script that can run in the background, continuously reading from a ZeroMQ stream of data and emitting it's own ZMQ stream of reconstructed results.

To make that happen, there are a few things I need to handle:

* I need to speed up the RPI reconstructions
* I probably will have to build ways for the reconstruction to process multiple files at once, because I suspect each reconstruction won't be able to fully tax even one small GPU
* I need to define a standard format for the calibration data that can easily be fed to the tool so it knows how to proceed.

