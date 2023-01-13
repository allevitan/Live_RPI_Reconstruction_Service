import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="live_rpi_reconstruction_service",
    version="0.1.0",
    python_requires='>3.7', # recommended minimum version for pytorch
    author="Abe Levitan",
    author_email="alevitan@mit.edu",
    description="A reconstruction engine for RPI data which runs on ZMQ and has an attached GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allevitan/Live_RPI_Reconstruction_Service",
    install_requires=[
        "numpy>=1.0",
        "scipy",
        "PyQt5",
        "pyzmq",
        "torch>=1.9.0", #1.9.0 supports autograd on indexed complex tensors
    ],
    entry_points={
        'console_scripts': [
            'lrrs=live_rpi_reconstruction_service.__main__:main',
            'stitch_rpi=live_rpi_reconstruction_service.stitch_rpi:main',
            'watch_live_rpi=live_rpi_reconstruction_service.watch_live_rpi:main',
            'watch_live_stitched_rpi=live_rpi_reconstruction_service.watch_live_stitched_rpi:main'
        ]
    },

    package_dir={"": "src"},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

