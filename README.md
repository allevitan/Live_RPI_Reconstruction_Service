Live RPI Reconstruction Service
-------------------------------

This project is a reconstruction engine for RPI data which runs on ZMQ and has an attached GUI. The reconstruction is done via a conjugate-gradient based method. The stepsize is chosen using a simple formula to approximate the optimal step size, which makes the algorithm tuning-parameter-free with only a minor penalty to performance.

This special reconstruction method is tuned for performance at the expense of some quality and flexibility, as is appropriate for a tool designed to provide live reconstructions to enable online analysis of data.

The reconstruction service itself reads in data and probe calibrations on two separate ZMQ streams, does the reconstructions on a selectable pool of GPUs, and then emits reconstructed results on a third ZMQ stream. The simple attached Qt5 GUI allows a user to monitor some basic metrics for the service, start and stop the service, and change the reconstruction parameters.

In addition to the central RPI reconstruction code, an accessory program is included (without a GUI) which can stitch together the output of the RPI reconstruction program into a single large-area scan. There are also two bare-bones command line scripts set up to inspect the output of the programs, if you are working in a situation where pystxmcontrol is broken or not installed.

## Installation

Once pytorch is installed, PCW and the remaining dependencies can be installed via:

```console
$ pip install -e .
```

The "-e" flag for developer-mode installation is recommended so updates to the git repo can be immediately included in the installed program.

## Usage

To run the main GUI program, simply run:

```console
$ lrrs
```

The RPI stitching program can be run with

```console
$ stitch_rpi
```

The individual frames output by the RPI reconstruction service can be inspected live using

```console
$ watch_live_rpi
```

And the stitched output can be watched using

```console
$ watch_live_stitched_rpi
```

## Configuration

All the ZMQ ports used by the various programs exposed by this package can be set in an optional configuration file. An example configuration file is found in "example_config.json". To override the default configuration, place a derived file called "config.json" in the main code folder, "src/live_rpi_reconstruction_service".

If the package is installed normally, you will need to set the configuration file before installing, and reinstall the package whenever you update the configuration. If the package is installed in developer mode, changes to the configuration file will be automatically propagated.


