from time import strftime
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import h5py
import readligo as r1
import matplotlib.mlab as mlab
# Goes across specified GPS times to find
# possible detections is SNR is above 8hz
# Assumes strain files are in the project directory
# Only works for one template at a time
# CODE MUST BE MODIFIED TO ACCOMODATE MULTIPLE TEMPLATE RUNS
# CHANGE BETWEEN H1 AND L1 AS NECESSARY

# create output file
gw_output = open('GWOutput6.txt', 'a')
listPossibleDetections = []  # SNRs of possible detections

# read the data file (16 seconds, sampled at 4096hz)
fs = 4096
# goes across multiple files to get good data based on given times
start = 931069952
stop = 931115008
gw_output.write('Real Time start: ' + str(strftime('%Y-%m-%d %H:%M:%S')) + '\n')
gw_output.write('GPS start: ' + str(start) + ' stop: ' + str(stop) + '\n')
gw_output.flush()
interval = 2048  # to prevent computer from running out of memory
newStart = start - interval
for i in range(0, int((stop - start) / interval)):
    newStart += interval
    newStop = newStart + interval

    #list of strain segments that statisfy CBC flag
    segList = r1.getsegs(newStart, newStop, 'L1', flag='CBCLOW_CAT1')
    tempStart = start

    for (begin, end) in segList:
        #uses the getstrain() method to load the data
        strain, meta, dq = r1.getstrain(begin, end, 'L1')

        time = np.arange(tempStart, tempStart + int(len(strain) / 4096.), 1. / fs)
        tempStart += int(len(strain) / 4096.)

        # read the template file (1 second, sampled at 4096hz)
        templateFile = h5py.File("rhOverM_Asymptotic_GeometricUnits.h5", "r")
        tempData = templateFile['Extrapolated_N2.dir/Y_l2_m-1.dat']
        tempStrain = tempData[:, 1]
        temp_time = np.arange(0, tempStrain.size / (1.0 * fs), 1. / fs)
        templateFile.close()
        
        # plt.figure()
        # plt.plot(time,strain)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Strain')

        #plt.figure()
        #plt.plot(temp_time, template)
        #plt.xlabel('Time (s)')
        #plt.ylabel('Strain')
        #plt.title('Template')
        # plt.show()

        # plots the amplitude spectral density of strain
        Pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=2 * fs)
        #plt.figure()
        #plt.loglog(freqs, np.sqrt(Pxx))
        #plt.axis([10, 2000, 1e-23, 1e-18])
        #plt.grid('on')
        #plt.xlabel('Freq (Hz)')
        #plt.ylabel('Strain / Hz$^{1/2}$')

        # plot ASD of the template
        power_data2, freq_psd2 = plt.psd(tempStrain, Fs=fs, NFFT=fs, visible=False)
        #plt.loglog(freq_psd2, np.sqrt(power_data2))
        #plt.xlabel('Frequency (Hz)')
        #plt.ylabel('ASD')
        # data displays amplitude spectral density
        # dip in plot of strain around 210Hz is sign of ringdown


        # apply bandpass filter between 80-250 Hz bc that
        # is where the detector noise is lowest
        (B, A) = sig.butter(4, [20 / (fs / 2.0), 400 / (fs / 2.0)], btype='pass')
        data_pass = sig.lfilter(B, A, strain)
        #plt.figure()
        # plt.plot(time, data_pass)
        # plt.title('Bandpass filtered data')
        # plt.xlabel('Time(s)')


        # time domain cross-correlation
        # tests similarity of time series at any time lag
        correlated_raw = np.correlate(strain, tempStrain, 'valid')
        correlated_passed = np.correlate(data_pass, tempStrain, 'valid')
        #plt.figure()
        #plt.plot(np.arange(0, (correlated_raw.size * 1.) / fs, 1.0 / fs), correlated_raw)
        #plt.title('Time Domain Cross-correlation')
        #plt.xlabel('Offset between data and template (s)')
        #plt.figure()
        #plt.plot(np.arange(0, (correlated_passed.size * 1.) / fs, 1.0 / fs), correlated_passed)
        #plt.xlabel('Offset between data and template (s)')
        # time delay is between the strain data and filter start
        #plt.title('Band Passed Time Domain Cross-correlation')
        # look for peak, this is when start of template matches the data


        # Matched Filtering in frequency domain to
        # take advantage of low noise frequency bins to increase the SNR

        # Fourier transform the data
        data_fft = np.fft.fft(strain)

        # template and data must be the same length, so pad template with zeros
        zero_pad = np.zeros(strain.size - tempStrain.size)
        template_pad = np.append(tempStrain, zero_pad)
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
        # plt.figure()
        # plt.plot(time, SNR)
        # plt.title('Matched Filter Output')
        # plt.xlabel('Time(s)')
        # plt.ylabel('SNR')
        # Look for peak of matched filter output, which is SNR of signal

        listMax1hz = []  # list of maximum SNR values per second
        SNRindx = 0
        print 'len(SNR): ', len(SNR)
        gw_output.write('len(SNR): ' + str(len(SNR)) + '\n')
        gw_output.flush()
        while SNRindx + 4096 < len(SNR):
            listMax1hz.append(np.amax(SNR[SNRindx:SNRindx + 4096]))
            SNRindx += 4096

        #gw_output.write('list of max SNRs: ' + str(listMax1hz) + '\n')

        for x in listMax1hz:
            if x > 8:
                listPossibleDetections.append(x)

        gw_output.write('possible detections: ' + str(listPossibleDetections) + '\n')
        gw_output.flush()

        # find times for possible detections
        detectTimesElems = []  # times of possible detections
        for x in range(0, len(time)):
            for y in listPossibleDetections:
                if abs(SNR[x] - y) < .0001:
                    detectTimesElems.append(x)

        for x in range(0, len(detectTimesElems)):
            detectTimesElems[x] = tempStart + (detectTimesElems[x] / (fs * 1.0))

        gw_output.write(str(strftime('%Y-%m-%d %H:%M:%S')) + '   possible times: ' + str(detectTimesElems) + '\n')
        gw_output.flush()

gw_output.write('--------------------------------------------------------------------------' + '\n')














