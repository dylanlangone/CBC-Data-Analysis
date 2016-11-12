#Plots Amplitude Spectral Density and Matched Filter output for specified times
#plug in start and stop times to analyze overlaps between detectors
#run twice: once for each detector
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import h5py
import readligo as r1
import matplotlib.mlab as mlab
import os, urllib
from time import strftime

gw_output = open('Plots.txt', 'a')
fs = 4096
# goes across multiple files to get good data based on given times
"""
First section lists the files
"""

import json, urllib
dataset = 'S6'

#CHANGE START/STOP TIMES HERE
start = 931186680
stop = 931186699


#CHANGE DETECTOR HERE
detector = 'H1'


urlformat = 'https://losc.ligo.org/archive/links/{0}/{1}/{2}/{3}/json/'
url = urlformat.format(dataset, detector, start, stop)
print "Tile catalog URL is ", url

r = urllib.urlopen(url).read()    # get the list of files
tiles = json.loads(r)             # parse the json

print tiles['dataset']
print tiles['GPSstart']
print tiles['GPSend']

output_list = open('plot_files', 'w')
for file in tiles['strain']:
    if file['format'] == 'hdf5':
        print "found file from ", file['UTCstart']
        print>>output_list, file['url']

output_list.close()

for line in open('plot_files').readlines():
    # Now build the URL for each file
    # Example line 815792128/H-H1_LOSC_4_V1-815886336-4096.hdf5
    tok = line.strip().split('/')
    url = line.strip()
    filename = tok[7]
    print 'filename: ', filename

    tol = filename.split('-')  # H   L1_LOSC_4_V1   931...   4096.hdf5
    # print tok
    # print tol
    gpstime = int(tok[6])

    print "Fetching data file from ", url
    r = urllib.urlopen(url).read()
    f = open(filename, 'w')  # write it to the right filename
    f.write(r)
    f.close()


    #start = h5py.File(filename, 'r')['meta']['GPSstart'].value
    #stop = h5py.File(filename, 'r')['meta']['Duration'].value + start
    #print 'GPSstart: ', start

    #newStart = start - interval
    #print 'initial newStart: ', newStart
    #for i in range(0, int((stop - start) / interval)):

        #newStart += interval
        #newStop = newStart + interval
        #print 'newStart: ', newStart, ' newStop: ', newStop
        # list of strain segments that statisfy CBC flag
    segList = r1.getsegs(start, stop, 'H1', flag='CBCLOW_CAT1')

    for (begin, end) in segList:
        # uses the getstrain() method to load the data
        strain, meta, dq = r1.getstrain(begin, end, 'H1')
        ts = 1 / 4096.
        #tempStart = meta['start']
        #tempStop = meta['stop']

        #print 'tempStart: ', tempStart
        time = np.arange(start, stop, ts)

        # read the template file (1 second, sampled at 4096hz)
        templateFile = h5py.File("rhOverM_Asymptotic_GeometricUnits.h5", "r")
        tempData = templateFile['Extrapolated_N2.dir/Y_l2_m-1.dat']
        templateStrain = tempData[:, 1]
        temp_time = np.arange(0, templateStrain.size / (1.0 * fs), 1. / fs)
        templateFile.close()

        # plots the amplitude spectral density of strain
        Pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=2 * fs)
        plt.figure()
        plt.loglog(freqs, np.sqrt(Pxx))
        #plt.axis([10, 2000, 1e-23, 1e-18])
        plt.grid('on')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Strain / Hz$^{1/2}$')

        # plot ASD of the template
        power_data2, freq_psd2 = plt.psd(templateStrain, Fs=fs, NFFT=fs, visible=False)
        #plt.figure()
        plt.loglog(freq_psd2, np.sqrt(power_data2), 'r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ASD')

        # apply bandpass filter between 80-250 Hz bc that
        # is where the detector noise is lowest
        (B, A) = sig.butter(4, [20 / (fs / 2.0), 250 / (fs / 2.0)], btype='pass')
        data_pass = sig.lfilter(B, A, strain)
        # plt.figure()
        # plt.plot(time, data_pass)
        # plt.title('Bandpass filtered data')
        # plt.xlabel('Time(s)')


        # time domain cross-correlation
        # tests similarity of time series at any time lag
        correlated_raw = np.correlate(strain, templateStrain, 'valid')
        correlated_passed = np.correlate(data_pass, templateStrain, 'valid')
        # plt.figure()
        # plt.plot(np.arange(0, (correlated_raw.size * 1.) / fs, 1.0 / fs), correlated_raw)
        # plt.title('Time Domain Cross-correlation')
        # plt.xlabel('Offset between data and template (s)')
        # plt.figure()
        # plt.plot(np.arange(0, (correlated_passed.size * 1.) / fs, 1.0 / fs), correlated_passed)
        # plt.xlabel('Offset between data and template (s)')
        # time delay is between the strain data and filter start
        # plt.title('Band Passed Time Domain Cross-correlation')
        # look for peak, this is when start of template matches the data


        # Matched Filtering in frequency domain to
        # take advantage of low noise frequency bins to increase the SNR

        # Fourier transform the data
        data_fft = np.fft.fft(strain)

        # template and data must be the same length, so pad template with zeros
        if (strain.size >= templateStrain.size):
            zero_pad = np.zeros(strain.size - templateStrain.size)
            template_pad = np.append(templateStrain, zero_pad)
            template_fft = np.fft.fft(template_pad)

            # Calculate the PSD of the data
            power_data, freq_psd = plt.psd(strain[12 * fs:], Fs=fs, NFFT=fs, visible=False)

            # Interpolate to get the PSD values at the needed frequencies
            datafreq = np.fft.fftfreq(strain.size) * fs
            power_vec = np.interp(datafreq, freq_psd, power_data)

            # calculate matched filter output
            optimal = data_fft * template_fft.conjugate() / power_vec
            # inverse fourier transform puts it back into time domain
            optimal_time = 2 * np.fft.ifft(optimal)

            # Normalize the matched filter output
            # SNR is 1 if there is just noise
            df = np.abs(datafreq[1] - datafreq[0])
            sigmasq = 2 * (template_fft * template_fft.conjugate() / power_vec).sum() * df
            sigma = np.sqrt(np.abs(sigmasq))
            SNR = abs(optimal_time) / (sigma)

            # plot SNR vs time
            plt.figure()
            plt.plot(time, SNR)
            plt.title('Matched Filter Output')
            plt.xlabel('Time(s)')
            plt.ylabel('SNR')
            plt.show()
            # Look for peak of matched filter output, which is SNR of signal

            listPossibleTimes = []
            listPossibleDetections = []  # SNRs of possible detections
            listMax1hz = []  # list of maximum SNR values per second
            SNRindx = 0
            print 'len(SNR): ', len(SNR)
            gw_output.write('len(SNR): ' + str(len(SNR)) + '\n')
            gw_output.flush()
            while SNRindx + 4096 < len(SNR):
                listMax1hz.append(np.amax(SNR[SNRindx:SNRindx + 4096]))
                SNRindx += 4096

            # gw_output.write('list of max SNRs: ' + str(listMax1hz) + '\n')

            for x in listMax1hz:
                if x > 8:
                    listPossibleDetections.append(x)

            gw_output.write('possible detections: ' + str(listPossibleDetections) + '\n')
            gw_output.flush()

            # find times for possible detections
            tempDetectTimesElems = []  # times of possible detections
            for x in range(0, len(time)):
                for y in listPossibleDetections:
                    if abs(SNR[x] - y) < .0001:
                        tempDetectTimesElems.append(x)

            for x in range(0, len(tempDetectTimesElems)):
                tempDetectTimesElems[x] = start + (tempDetectTimesElems[x] * ts)

            for x in tempDetectTimesElems:
                listPossibleTimes.append(x)
            gw_output.write(str(strftime('%Y-%m-%d %H:%M:%S')) + '   possible times: ' + str(listPossibleTimes) + '\n')
            gw_output.flush()

    os.remove(filename)