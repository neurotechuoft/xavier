'''
This script can stream data from a board and plot the EEG data as a time series in real-time, as well as 
plot the PSD and band plot in real-time.
Goals: classify data at the same time.
'''

import argparse
import logging

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, AggOperations, NoiseTypes, WindowFunctions
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

from pyqtgraph.Qt import QtGui, QtCore

import time
import datetime

import matplotlib
import numpy as np
import pandas as pd

import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Object to graphing time series in real-time
class Graph:
    def __init__(self, board_shim):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_pens(self):
        self.pens = list()
        self.brushes = list()
        colors = ['#A54E4E', '#A473B6', '#5B45A4', '#2079D2', '#32B798', '#2FA537', '#9DA52F', '#A57E2F', '#A53B2F']
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    # For real-time PSD
    def _init_psd(self):
        self.psd_plot = self.win.addPlot(row=0, col=1, rowspan=len(self.exg_channels) // 2)
        self.psd_plot.showAxis('left', False)
        self.psd_plot.setMenuEnabled('left', False)
        self.psd_plot.setTitle('PSD Plot')
        self.psd_plot.setLogMode(False, True)
        self.psd_curves = list()
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        for i in range(len(self.exg_channels)):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            psd_curve.setDownsampling(auto=True, method='mean', ds=3)
            self.psd_curves.append(psd_curve)

    # For real-time band plot
    def _init_band_plot(self):
        self.band_plot = self.win.addPlot(row=len(self.exg_channels) // 2, col=1, rowspan=len(self.exg_channels) // 2)
        self.band_plot.showAxis('left', False)
        self.band_plot.setMenuEnabled('left', False)
        self.band_plot.showAxis('bottom', False)
        self.band_plot.setMenuEnabled('bottom', False)
        self.band_plot.setTitle('BandPower Plot')
        y = [0, 0, 0, 0, 0]
        x = [1, 2, 3, 4, 5]
        self.band_bar = pg.BarGraphItem(x=x, height=y, width=0.8, pen=self.pens[0], brush=self.brushes[0])
        self.band_plot.addItem(self.band_bar)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())
            if data.shape[1] > self.psd_size:
                # plot psd
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2,
                                                    self.sampling_rate,
                                                    WindowFunctions.BLACKMAN_HARRIS.value)
                lim = min(70, len(psd_data[0]))
                self.psd_curves[count].setData(psd_data[1][0:lim].tolist(), psd_data[0][0:lim].tolist())
                # plot bands
                avg_bands[0] = avg_bands[0] + DataFilter.get_band_power(psd_data, 2.0, 4.0)
                avg_bands[1] = avg_bands[1] + DataFilter.get_band_power(psd_data, 4.0, 8.0)
                avg_bands[2] = avg_bands[2] + DataFilter.get_band_power(psd_data, 8.0, 13.0)
                avg_bands[3] = avg_bands[3] + DataFilter.get_band_power(psd_data, 13.0, 30.0)
                avg_bands[4] = avg_bands[4] + DataFilter.get_band_power(psd_data, 30.0, 50.0)

        avg_bands = [int(x * 100 / len(self.exg_channels)) for x in avg_bands]
        self.band_bar.setOpts(height=avg_bands)

        self.app.processEvents()


def main():
    # Get user input on board type and if user wants to classify stream
    is_synthetic_board = int(input("Enter 1 to stream from a synthetic board, 0 otherwise: "))
    if is_synthetic_board == 1:
        is_synthetic_board = True
    elif is_synthetic_board == 0:
        is_synthetic_board = False
    is_classify = int(input("Enter 1 to include classification in the pipeline, 0 otherwise: "))
    if is_classify == 1:
        is_classify = True
    elif is_classify == 0:
        is_classify = False
        
    BoardShim.enable_dev_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    # If streaming from a real board
    if not is_synthetic_board:
        logging.basicConfig(level=logging.DEBUG)

        parser = argparse.ArgumentParser()
        # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=0)
        parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
        parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                            default=0)
        parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
        parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
        parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=False, default=BoardIds.SYNTHETIC_BOARD)
        parser.add_argument('--file', type=str, help='file', required=False, default='')
        parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                            required=False, default=BoardIds.NO_BOARD)
        parser.add_argument('--preset', type=int, help='preset for streaming and playback boards',
                            required=False, default=BrainFlowPresets.DEFAULT_PRESET)
        args = parser.parse_args()

        params = BrainFlowInputParams()
        params.ip_port = args.ip_port
        params.serial_port = args.serial_port
        params.mac_address = args.mac_address
        params.other_info = args.other_info
        params.serial_number = args.serial_number
        params.ip_address = args.ip_address
        params.ip_protocol = args.ip_protocol
        params.timeout = args.timeout
        params.file = args.file
        params.master_board = args.master_board
        params.preset = args.preset
    else:
        # Synthetic board
        params = BrainFlowInputParams()

    try:
        if not is_synthetic_board:
            board_shim = BoardShim(args.board_id, params)
        print("Initiating")
        if is_synthetic_board:
            board_id = BoardIds.SYNTHETIC_BOARD.value
            board_shim = BoardShim(board_id, params)
        print("Preparing session...")
        board_shim.prepare_session()
        # For non-synthetic board
        if not is_synthetic_board:
            board_shim.start_stream(450000, args.streamer_params)
        print("Starting stream at", time.strftime("%H:%M:%S"))
        board_shim.start_stream()
        print("Stream was started")
        # Plot time series in real-time
        print("Starting real-time graphing...")
        Graph(board_shim)
        print("Done plotting time series\n")

        # Signal processing pipeline
        print("=== Starting processing pipeline ===")
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        # time.sleep(10)
        if is_classify:
            time.sleep(5)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
        data = board_shim.get_board_data()
        print(data)
        print("Stopping stream and releasing session")
        board_shim.stop_stream()
        print("Ended stream at", time.strftime("%H:%M:%S"))
        board_shim.release_session()

        # demo how to convert it to pandas DF and plot data
        print("Getting EEG channels")
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        df = pd.DataFrame(np.transpose(data))
        plt.figure()
        print("Plotting before-processing data")
        # print("Data:")
        # print(df[eeg_channels])
        df[eeg_channels].plot(subplots=True)
        os.chdir("plots/")
        plt.savefig('before_processing.png')
        os.chdir("../")

        # for demo apply different filters to different channels, in production choose one
        print("Processing: applying filters")
        for count, channel in enumerate(eeg_channels):
            # filters work in-place
            if count == 0:
                DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(board_id), 2.0, 50.0, 4,
                                            FilterTypes.BESSEL.value, 0)
            elif count == 1:
                DataFilter.perform_bandstop(data[channel], BoardShim.get_sampling_rate(board_id), 48.0, 52.0, 3,
                                            FilterTypes.BUTTERWORTH.value, 0)
            elif count == 2:
                DataFilter.perform_lowpass(data[channel], BoardShim.get_sampling_rate(board_id), 50.0, 5,
                                        FilterTypes.CHEBYSHEV_TYPE_1.value, 1)
            elif count == 3:
                DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(board_id), 2.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)
            elif count == 4:
                DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
            else:
                DataFilter.remove_environmental_noise(data[channel], BoardShim.get_sampling_rate(board_id),
                                                    NoiseTypes.FIFTY.value)

        df = pd.DataFrame(np.transpose(data))
        plt.figure()
        print("Plotting after-processing data")        
        df[eeg_channels].plot(subplots=True)
        os.chdir("plots/")
        plt.savefig('after_processing.png')
        os.chdir("../")

        # TODO To make classification real-time, can try epoching the signal at small intervals and classifying data at epochs
        # Sliding window technique with a queue containing a certain duration of processed streaming data.

        if is_classify:
            print("Starting classification")
            master_board_id = board_shim.get_board_id()
            sampling_rate = BoardShim.get_sampling_rate(master_board_id)

            eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
            print("Getting average band powers")
            bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
            feature_vector = bands[0]
            print("feature_vector:", feature_vector)

            print("Setting up mindfulness params")
            # First arg: MINDFULNESS = 0, RESTFULNESS = 1, USER_DEFINED = 2
            # Second arg: DEFAULT_CLASSIFIER = 0, DYN_LIB_CLASSIFIER = 1, ONNX_CLASSIFIER = 2
            mindfulness_params = BrainFlowModelParams(0,0)
            print("mindfulness_params:", mindfulness_params)
            mindfulness = MLModel(mindfulness_params)
            mindfulness.prepare()
            print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
            mindfulness.release()

            print("Setting up restfulness params")
            # First arg: MINDFULNESS = 0, RESTFULNESS = 1, USER_DEFINED = 2
            # Second arg: DEFAULT_CLASSIFIER = 0, DYN_LIB_CLASSIFIER = 1, ONNX_CLASSIFIER = 2
            restfulness_params = BrainFlowModelParams(1,0)
            restfulness = MLModel(restfulness_params)
            restfulness.prepare()
            print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
            restfulness.release()
            print("Done")

    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()



# From OpenBCI forum:
# Typically two types of DSP (digital signal processing) filters are applied to the raw data 
# stream before plotting: a bandpass filter from say .5 Hz to 45 Hz. 
# And a 'notch' filter at your local mains frequency (50 or 60 Hz).

'''
To process in real-time, we need to create epochs from the original raw data stream. Ideally, we want to overlap the epochs as much as possible
to approximate processing continuity. If the epochs are too far apart, then the BCI system will take too long to process an EEG signal and the command will take too long


In order for the device to classify correctly, we need to provide it training data, i.e. to 'calibrate' it before it can classify in real-time
This means the user will have to do things such that the device knows what each class looks like, and then it can begin associating processed EEG signals a class during classification

Once the model has been trained, we can start the real-time processing. This means streaming data and dividing it into epochs, collecting one epoch at a time, 
processing it, and feeding it to the model which should immediately give an answer as to what class the signal belongs to.
'''