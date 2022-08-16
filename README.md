Live RPI Reconstruction Service
-------------------------------

This project is a reconstruction engine for RPI data which runs on ZMQ and has an attached GUI. The reconstruction is done via a conjugate-gradient based method with a stepsize that is intelligently chosen with an analytic formula that approximates the optimal step to maximally reduce the mean squared amplitude error.

This special reconstruction method is tuned for performance at the expense of some quality and flexibility, as is appropriate for a tool designed to provide live reconstructions to enable online analysis of data.

The reconstruction service itself reads in data and probe calibrations on two separate ZMQ streams, does the reconstructions on a selectable pool of GPUs, and then emits reconstructed results on a third ZMQ stream. The simple attached Qt5 GUI allows a user to monitor some basic metrics for the service, start and stop the service, and change the reconstruction parameters.

## Installation

Once pytorch is installed, PCW and the remaining dependencies can be installed via:

```console
$ pip install -e .
```

The "-e" flag for developer-mode installation is recommended so updates to the git repo can be immediately included in the installed program.

## Usage

To run the program, simply run:

```console
$ lrrs
```
