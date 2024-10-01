from astropy.io import fits
import numpy as np
import scipy
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy import interpolate
from easyspec.cleaning import cleaning
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter
from astropy import units as u
from scipy.signal import medfilt
from dust_extinction.parameter_averages import F99

plt.rcParams.update({'font.size': 12})

libpath = Path(__file__).parent.resolve() / Path("airmass")
libpath_std = Path(__file__).parent.resolve() / Path("standards")
cleaning = cleaning()

easyspec_extraction_version = "1.0.0"


class extraction:

    """This class contains all the functions necessary to perform the spectral extraction from a cleaned/stacked image."""

    def __init__(self): 
        # Print the current version of easyspec-extraction       
        print("easyspec-extraction version: ",easyspec_extraction_version)

    def find_nearest(self,array, value):

        """For a monotonically increasing/decreasing array, this function finds the index of the element nearest to the given value
        
        Parameters
        ----------
        array: numpy array (float)
            Monotonically increasing/decreasing array.
        
        value: float
            Value to look for in the array.

        Returns
        -------
        idx: int
            The index corresponding to the closest element to the given value.
        
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def import_data(self, target_spec_file, target_name, exposure_target = None, exposure_header_entry="AVEXP", airmass_target = None, airmass_header_entry="AVAIRMAS", std_star_spec_file = None, exposure_std_star = None, airmass_std_star = None, plot = True):

        """
        This function opens the data file (i.e. fits image with one or more spectra) and reads the exposure time from its header. Optionally, the user can manually add the exposure time.
        If you want to calibrate your spectrum in flux, please also add the standard star spectral fits file and the reddening.

        Parameters
        ----------
        target_spec_file: string 
            The path to the data '.fits' or '.fit' file containing the target spectrum.
        target_name: string
            The name of the target to be used in subsequent plots.
        exposure_target: float
            Optional. You can manually pass the exposure time (in seconds) of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'.
        exposure_header_entry: string
            Optional. If 'exposure_target' is not None, easyspec will automatically look into the image header for the exposure time corresponding to this entry. This entry will be used for the target and standard star images.
        airmass_target: float
            Optional. You can manually pass the average airmass of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'airmass_header_entry'.
        airmass_header_entry: string
            Optional. If 'airmass_target' is not None, easyspec will automatically look into the image header for the average airmass corresponding to this entry. This entry will be used for the target and standard star images.
        std_star_spec_file: string
            Optional. The path to the data '.fits' or '.fit' file containing the standard star spectrum.
        exposure_std_star: float
            Optional. You can manually pass the exposure time (in seconds) of your observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'exposure_header_entry'
        airmass_std_star: float
            Optional. You can manually pass the average airmass of your standard star observation. If 'None', easyspec will automatically look in the image header for the entry given by the variable 'airmass_header_entry'.        
        plot: boolean
            If True, easyspec will plot the input spectra.


        Returns
        -------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        std_star_spec_data: numpy.ndarray (float)
            Matrix containing the standard star spectral image (only if the standard star spectral file is given by the user).
        """

        # Main target:
        self.target_spec_data = fits.open(target_spec_file)[0].data
        self.target_name = target_name
        if exposure_target is None:
            self.exposure_target = cleaning.look_in_header(target_spec_file,exposure_header_entry)
        else:
            self.exposure_target = exposure_target
        if airmass_target is not None:
            self.airmass_target = airmass_target
        elif airmass_header_entry is not None:
            self.airmass_target = cleaning.look_in_header(target_spec_file,airmass_header_entry)

        self.image_shape = np.shape(self.target_spec_data)
        self.aspect_ratio = self.image_shape[0]/self.image_shape[1]

        # Standard star:
        if std_star_spec_file is not None:
            self.std_star_spec_data = fits.open(std_star_spec_file)[0].data
            if exposure_std_star is not None:
                self.exposure_std_star = exposure_std_star
            elif std_star_spec_file is not None:
                self.exposure_std_star = cleaning.look_in_header(std_star_spec_file,exposure_header_entry)
            if airmass_std_star is not None:
                self.airmass_std_star = airmass_std_star
            elif airmass_header_entry is not None:
                self.airmass_std_star = cleaning.look_in_header(std_star_spec_file,airmass_header_entry)

        if plot:
            cleaning.plot(image_data=self.target_spec_data,figure_name=target_name,save=False)
            if std_star_spec_file is not None:
                cleaning.plot(image_data=self.std_star_spec_data,figure_name="Standard star",save=False)

        if std_star_spec_file is not None:
            print("Returning both target and std star images...")
            return self.target_spec_data, self.std_star_spec_data
        else:
            print("No standard star file was found. Returning only the target image.")
            return self.target_spec_data
        
    def tracing(self, target_spec_data, method = "argmax", y_pixel_range = 15, xlims = None, poly_order = 2, trace_half_width = 7, peak_height = 100, distance = 50, Number_of_slices = 20, peak_dispersion_limit = 3, main_plot = True, plot_residuals = True):

        """
        This function detects all the spectra available in an image and fits a polynomial to recover the trace of each one of them. 
        It can be used to recover the spectral spine (or trace) of one or several spectra in the same image.

        Parameters
        ----------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        method: string
            The method to be adopted in finding the trace. Options are "argmax" or "moments" if you want to extract only the strongest spectrum in the image, or "multi" if you
            want to extract more than one spectrum. The option "argmax" works very well although is limited by the pixel size. The option "moments" is suited only for spectra 
            with a very high signal-to-noise ratio. The option "multi" will use the "argmax" method to look for several spectra in a single image.
        y_pixel_range: integer
            Y-axis range of pixels to be used to extract a specific spectrum. The total y-axis interval adopted is 2*y_pixel_range.
        xlims: list or array
            A list/array with the two limits in the x-axis delimiting the region where we will look for the spectra. E.g. if your image x-axis has 2000 pixels but your spectrum
            goes from pixels ~700 up to pixel ~1800, you can set xlims = [700,1800] to avoid fitting regions of pure noise. This variable is useful only for the "argmax" or
            "moments" methods.
        poly_order: integer
            The order of the polynomial(s) used to fit the trace.
        trace_half_width: integer
            The half-width of the trace. Here it is useful for estimating the background and plotting the data.
        peak_height: float
            The required height of the peaks. The smaller this value, the more spectra you will find in your image.
        distance: float
            Required minimal distance (>= 1) in pixels between neighbouring peaks in the y-axis. Default is 20 pixels. If you set this value too high with respect to the y-axis
            resolultion of your image, this function will return undesired results.
        Number_of_slices: integer
            Useful only for the "multi" mode. This is the number of slices in which every spectra in the image is divided in the x-axis. It is used to set the best x-axis interval
            at which easyspec will fit the trace.
        peak_dispersion_limit: float
            Useful only for the "multi" mode. This parameter sets the dispersion limit (with respect to the trace slice of minimum dispersion) of a trace slice such that we can
            use this slice to fit the trace, i.e. if the trace dispersion in a slice is larger than peak_dispersion_limit*minimum_dispersion, this slice will be ignored in the
            fit, where the term 'minimum_dispersion' stands for the dispersion in the slice with minimum trace dispersion.
        main_plot: boolean
            If True, easyspec will plot all the traces over all the spectra (or the main spectra) in the image.
        plot_residuals: boolean
            If True, easyspec will plot the residuals for each trace fitted to the data.
        
        Returns
        -------
        fitted_polymodel_list: list
            List with the fitted spine for all spectra in the image or only the brighter spectrum.
        """

        try:
            _ = self.aspect_ratio
        except:
            raise NameError("The variable self.aspect_ratio does not exist. Please load your data with the function extraction.import_data() to avoid this issue.")

        polymodel = Polynomial1D(degree = poly_order)
        linfitter = LinearLSQFitter()

        yvals = []
        yvals.append( np.argmax(target_spec_data, axis=0) ) # Generates an array containing only the pixels with the maximum number of counts for each column
        median_trace_value = np.median(yvals[0])
        xvals = np.arange(np.shape(target_spec_data)[1])
        ymin = []
        ymin.append(int(median_trace_value - y_pixel_range))
        ymax = []
        ymax.append(int(median_trace_value + y_pixel_range))

        if xlims is None:
            xmin, xmax = 0, np.shape(target_spec_data)[1]
        else:
            xmin, xmax = xlims[0], xlims[1]
        
        fitted_polymodel_list = []
        bad_pixels = []
        if method == "argmax":
            bad_pixels.append( (yvals[0] < ymin[0]) | (yvals[0] > ymax[0]) | (xvals < xmin) | (xvals > xmax) )  # That "|" is a bitwise "or" with assignment
            fitted_polymodel_list.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[0][~bad_pixels[-1]]))
            final_peak_positions = [median_trace_value]
        
        elif method == "moments":
            yvals = []
            yaxis = np.repeat(np.arange(ymin[0], ymax[0])[:,None], target_spec_data[:,:].shape[1], axis=1)


            background1 = [] 
            background2 = []
            for i in range(np.shape(target_spec_data)[1]):
                background1.append(np.median(target_spec_data[int(median_trace_value)+trace_half_width:ymax[0],i]))
                background2.append(np.median(target_spec_data[ymin[0]:int(median_trace_value)-trace_half_width,i]))

            background1 = np.asarray(background1*np.shape(target_spec_data)[0])  # Here we increase the size of the list by a factor equal to the y-axis resolution and transform it into an array
            background2 = np.asarray(background2*np.shape(target_spec_data)[0])
            background1 = background1.reshape(np.shape(target_spec_data))  # Each *colum* of this matrix is filled with the same background value
            background2 = background2.reshape(np.shape(target_spec_data))
            background = (background1+background2)/2
                
            # moment 1 is the data-weighted average of the Y-axis coordinates
            weight = target_spec_data[ymin[0]:ymax[0],:] - background[ymin[0]:ymax[0],:]
            negative_vals = weight < 0.0   #The signal must be zero in the signal-free region for m1 to return an accurate estimate of the location of the peak
            weight[negative_vals] = 0.0 # Padding the background negative values with zeros
            yvals.append( np.average(yaxis, axis=0, weights=weight) )
            
            bad_pixels.append( (yvals[0] > ymax[0]) | (yvals[0] < ymin[0]) | (xvals < xmin) | (xvals > xmax) )
            fitted_polymodel_list.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[0][~bad_pixels[-1]]))
            final_peak_positions = [median_trace_value]
        
        elif method == "multi":
            yvals = []
            multi_peaks = []
            for i in range(len(target_spec_data[0,:])):
                local_peaks = scipy.signal.find_peaks(target_spec_data[:,i],height=peak_height,distance=distance)[0]
                multi_peaks.append(local_peaks)
            
            multi_peaks = np.asarray(multi_peaks,dtype=object)
            flat_multi_peaks = multi_peaks[0]
            for i in multi_peaks[1:]:
                flat_multi_peaks = np.concatenate([flat_multi_peaks,i])
            
            peaks_histogram = np.histogram(flat_multi_peaks, bins=np.shape(target_spec_data)[0])
            final_peak_positions = scipy.signal.find_peaks(peaks_histogram[0],height=peak_height,distance=distance)[0]
            print("Total number of spectra: ", len(final_peak_positions),", centered at y-pixels ", final_peak_positions)

            fitted_polymodel_list = []
            ymin = []
            ymax = []
            # The purpose of this loop is to find the minimum and maximum values in the x-axis at which it is still worth to use the peak
            for peak in final_peak_positions:
                ymin.append(int(peak - y_pixel_range))
                ymax.append(int(peak + y_pixel_range))
                yvals.append( np.argmax(target_spec_data[ymin[-1]:ymax[-1]], axis=0) + ymin[-1] )
                
                peak_dispersion = []
                spec_slices = np.linspace(0,target_spec_data.shape[1],Number_of_slices)
                interval = int(np.diff(spec_slices)[0])
                x_cut = []
                for i in spec_slices[:-1]:
                    i = int(i)
                    slice_dispersion = np.std(yvals[-1][i:i+interval])
                    if slice_dispersion < 1:
                        slice_dispersion = 1
                    peak_dispersion.append(slice_dispersion)
                    x_cut.append(xvals[i+interval])
                
                mininum_dispersion = np.asarray(peak_dispersion).min()
                for n,dispersion in enumerate(peak_dispersion):
                    if dispersion > peak_dispersion_limit*mininum_dispersion:
                        xmin = x_cut[n]
                    else:
                        break
                
                for n,dispersion in enumerate(np.flip(peak_dispersion)):
                    if dispersion > peak_dispersion_limit*mininum_dispersion:
                        xmax = x_cut[-n-2]
                    else:
                        break

                bad_pixels.append( (yvals[-1] < ymin[-1]) | (yvals[-1] > ymax[-1]) | (xvals < xmin) | (xvals > xmax))
                # We fit a polynomial to extract the trace:
                fitted_polymodel_list.append(linfitter(polymodel, xvals[~bad_pixels[-1]], yvals[-1][~bad_pixels[-1]]))
        else:
            raise RuntimeError("Wrong method name. Accepted entries: 'argmax', 'moments' or 'multi'.")

        if main_plot:
            plt.figure(figsize=(12,12*self.aspect_ratio)) 
            plt.imshow(np.log10(target_spec_data), vmax=np.log10(target_spec_data.max()), cmap = "gray", origin='lower')
            plt.gca().set_aspect(1)
            if len(final_peak_positions) > 1:
                plt.title(f"We have a total of {len(final_peak_positions)} detected spectra")
            else:
                plt.title(f"We have detected {len(final_peak_positions)} spectrum")
            for n in range(len(fitted_polymodel_list)):
                plt.plot(xvals, fitted_polymodel_list[n](xvals), 'C0')
                plt.fill_between(xvals, fitted_polymodel_list[n](xvals)-trace_half_width, fitted_polymodel_list[n](xvals)+trace_half_width, color='C1', alpha=0.3)
                plt.text(target_spec_data.shape[1]*0.1, final_peak_positions[n], f"Spec {n}", color="limegreen", fontsize=14)

        if plot_residuals:
            for n in range(len(fitted_polymodel_list)):
                plt.figure(figsize=(12,12*self.aspect_ratio)) 
                plt.imshow(np.log10(target_spec_data[ymin[n]:ymax[n],:]), extent=[0,target_spec_data.shape[1],ymin[n],ymax[n]], vmax=np.log10(target_spec_data[ymin[n]:ymax[n],:].max()), cmap = "gray", origin='lower')
                plt.gca().set_aspect(10)
                plt.plot(xvals, fitted_polymodel_list[n](xvals), 'C0')
                plt.axis((0,len(target_spec_data[0,:]),ymin[n],ymax[n]))
                plt.fill_between(xvals, fitted_polymodel_list[n](xvals)-trace_half_width, fitted_polymodel_list[n](xvals)+trace_half_width, color='C1', alpha=0.15)
                plt.title(f"Spectrum {n} - "+self.target_name+" field")

                plt.figure(figsize=(12,5))
                plt.plot(xvals[~bad_pixels[n]], yvals[n][~bad_pixels[n]] - fitted_polymodel_list[n](xvals[~bad_pixels[n]]), 'x')
                plt.plot([xvals.min(),xvals.max()],[0,0],"k--")
                plt.ylabel("Trace residuals (data-model)")
                plt.title(f"Spectrum {n} - Trace residuals")

        if main_plot or plot_residuals:
            plt.show()

        return fitted_polymodel_list
    
    
    def extracting(self, target_spec_data, fitted_polymodel_list, master_lamp_data = None, trace_half_width = 7, shift_y_pixels = 30, lamp_peak_height = None, peak_distance = None, diagnostic_plots = True, spec_plots = True):

        """
        This function takes the traces as input to extract one or more spectra from the image with a Gaussian-weighted model.

        In case you need references for the lamp peaks, here are some databases you can use (as of 12 September, 2024):
         * TNG DOLORES spectrograph reference lamps can be found in table 1 here: https://www.tng.iac.es/instruments/lrs/
         * SOAR Goodman Spectrograph Reference Lamp Library: https://soardocs.readthedocs.io/projects/lamps/en/latest/
         * Observatorio do Pico dos Dias (OPD-LNA) spectrographs: https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd/instrumentacao/instrumentos-e-detectores

        Parameters
        ----------
        target_spec_data: numpy.ndarray (float)
            Matrix containing the target spectral image.
        fitted_polymodel_list: string
            List with the fitted spine for one or more spectra in the target spectral image. This is the output of the function extraction.tracing().
        master_lamp_data: string
            Optional. The path to the master lamp data file. This file can be produced with the easyspec cleaning() class.
        trace_half_width: integer
            The half-width of the trace. The spectra will be extracted within a region 2*trace_half_width around their corresponding traces.
        shift_y_pixels: integer
            Value used to shift the trace vertically by +-N pixels and repeat the extracting process to get the sky spectrum around each trace.
        lamp_peak_height: float
            The height of the peaks (in counts and with respect to zero) to be detected.
        peak_distance: float
            The minimum distance between peaks in pixels. Default is the length of the x-axis divided by 20.
        diagnostic_plots: boolean
            If True, easyspec will plot the spectral images overlaid with the target's spectral trace and corresponding sky traces.
        spec_plots: boolean
            If True, easyspec will plot the extracted spectra for the target(s) and the lamp.
        
        Returns
        -------
        spec_list: list
            List with the extracted spectra of all the targets in the image or only the brightest spectrum.
        lamp_spec_list: list
            Optional. List with the extracted spectra of the lamp based on the traces of all spectra find in the image.
        lamp_peak_positions_list: list
            Optional. List with the positions (in pixels) of all peaks in the lamp spectra with heights above the spectrum median. We have one list with
            peak positions for each trace given as an input in the variable fitted_polymodel_list. Notice that the peak positions are slightly different
            for different traces.
        """

        try:
            _ = self.aspect_ratio
        except:
            raise NameError("The variable self.aspect_ratio does not exist. Please load your data with the function extraction.import_data() to avoid this issue.")


        individual_spec_shift = False
        if isinstance(shift_y_pixels, int):
            # The value shift_y_pixels must be at least the triple of trace_half_width, otherwise the background will be estimated too close to the spectral trace.
            if shift_y_pixels < 3*trace_half_width:
                print(f"The given value of 'shift_y_pixels' is below the limit '3*trace_half_width'. We are resetting shift_y_pixels to 3*trace_half_width (i.e. = {3*trace_half_width})")
                shift_y_pixels = 3*trace_half_width
            shift_y_pixels_up = shift_y_pixels
            shift_y_pixels_down = shift_y_pixels
        elif isinstance(shift_y_pixels, list):
            if np.shape(shift_y_pixels) == (2,):
                shift_y_pixels_up = int(shift_y_pixels[0])
                shift_y_pixels_down = int(shift_y_pixels[1])
            elif np.shape(shift_y_pixels)[0] == len(fitted_polymodel_list):
                individual_spec_shift = True
            else:
                raise TypeError("Invalid input for shift_y_pixels. This parameter must be an integer, a list with 2 integers, or a list with N pairs of integers given as sublists, where N is equal to the length of fitted_polymodel_list.")
        else:
            raise TypeError("Invalid input for shift_y_pixels. This parameter must be an integer, a list with 2 integers, or a list with N pairs of integers given as sublists, where N is equal to the length of fitted_polymodel_list.")

        xvals = np.arange(np.shape(target_spec_data)[1])

        spec_list = []
        lamp_spec_list = []
        lamp_peak_positions_list = []
        lamp_peak_heights_list = []


        for number,fitted_polymodel in enumerate(fitted_polymodel_list):
            if individual_spec_shift:
                shift_y_pixels_up = shift_y_pixels[number][0]
                shift_y_pixels_down = shift_y_pixels[number][1]

            trace_center = fitted_polymodel(xvals)
            median_trace_value = np.median(trace_center)
            ymin = int(median_trace_value - shift_y_pixels_up)
            ymax = int(median_trace_value + shift_y_pixels_down)

            # Here we estimate the background in the regions around the trace, excluding the region within trace_half_width
            background1 = [] 
            background2 = []
            for i in range(np.shape(target_spec_data)[1]):
                background1.append(np.median(target_spec_data[int(median_trace_value)+trace_half_width:ymax,i]))
                background2.append(np.median(target_spec_data[ymin:int(median_trace_value)-trace_half_width,i]))

            background1 = np.asarray(background1*np.shape(target_spec_data)[0])  # Here we increase the size of the list by a factor equal to the y-axis resolution and transform it into an array
            background2 = np.asarray(background2*np.shape(target_spec_data)[0])
            background1 = background1.reshape(np.shape(target_spec_data))  # Each *colum* of this matrix is filled with the same background value
            background2 = background2.reshape(np.shape(target_spec_data))
            background = (background1+background2)/2

            # Here we select only the region around the spine:
            cutouts = np.array([target_spec_data[int(yval)-trace_half_width:int(yval)+trace_half_width, xval]
                        for yval, xval in zip(trace_center, xvals)])
            
            mean_trace_profile = (cutouts - background.T[:,:np.shape(cutouts)[1]]).mean(axis=0)  # Here we also transpose and crop the background matrix just to fit the size of "cutouts"
            trace_profile_xaxis = np.arange(-trace_half_width, trace_half_width)

            lmfitter = LevMarLSQFitter()
            guess = Gaussian1D(amplitude=mean_trace_profile.max(), mean=0, stddev=trace_half_width/2)
            fitted_trace_profile = lmfitter(model=guess, x=trace_profile_xaxis, y=mean_trace_profile)
            model_trace_profile = fitted_trace_profile(trace_profile_xaxis)

            # Lamp spectral extraction:
            if master_lamp_data is not None:
                lamp_image = fits.open(master_lamp_data)
                lamp_image = lamp_image[0].data
                lamp_spectrum = np.array([np.average(lamp_image[int(yval)-trace_half_width:int(yval)+trace_half_width, xval],
                                        weights=model_trace_profile) for yval, xval in zip(trace_center, xvals)])

                if lamp_peak_height is None:
                    lamp_peak_height = np.median(lamp_spectrum)
                if peak_distance is None:
                    peak_distance = int(len(xvals)/20)
                lamp_peak_positions, lamp_peak_heights = scipy.signal.find_peaks(lamp_spectrum,distance=peak_distance,height=lamp_peak_height)
                lamp_spec_list.append(lamp_spectrum)
                lamp_peak_positions_list.append(lamp_peak_positions)
                lamp_peak_heights_list.append(lamp_peak_heights)


            # Target spectral extraction:
            gaussian_trace_avg_spectrum = np.array([np.average(
            target_spec_data[int(yval)-trace_half_width:int(yval)+trace_half_width, xval] - background[int(yval)-trace_half_width:int(yval)+trace_half_width, xval],
            weights=model_trace_profile) for yval, xval in zip(trace_center, xvals)])

            # Sky spectra:
            trace_center_sky = fitted_polymodel(xvals) + shift_y_pixels_up - trace_half_width
            trace_center_sky2 = fitted_polymodel(xvals) - shift_y_pixels_down + trace_half_width
            
            gaussian_trace_sky_spectrum1 = np.array([np.average(
            target_spec_data[int(yval)-trace_half_width:int(yval)+trace_half_width, xval] - background[int(yval)-trace_half_width:int(yval)+trace_half_width, xval],
            weights=model_trace_profile) for yval, xval in zip(trace_center_sky, xvals)])

            gaussian_trace_sky_spectrum2 = np.array([np.average(
            target_spec_data[int(yval)-trace_half_width:int(yval)+trace_half_width, xval] - background[int(yval)-trace_half_width:int(yval)+trace_half_width, xval],
            weights=model_trace_profile) for yval, xval in zip(trace_center_sky2, xvals)])

            gaussian_trace_sky_spectrum = (gaussian_trace_sky_spectrum1 + gaussian_trace_sky_spectrum2)/2  # Final sky spectrum
            
            # Final target spectrum:
            gaussian_trace_avg_spectrum = gaussian_trace_avg_spectrum - gaussian_trace_sky_spectrum

            spec_list.append(gaussian_trace_avg_spectrum)

            if spec_plots:
                plt.figure(figsize=(12,5)) 
                plt.plot(gaussian_trace_avg_spectrum, label=f'Gaussian trace spec {number}', alpha=0.9, linewidth=0.5, color='orange')
                plt.title(f"Non-calibrated Gaussian-extracted spectrum {number} - "+self.target_name+" field")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.ylabel("Counts")
                plt.xlabel("Pixels")
                plt.legend(loc='upper left')

                if master_lamp_data is not None:
                    plt.figure(figsize=(12,5)) 
                    plt.plot(lamp_spectrum, label='Lamp spec', alpha=0.9, linewidth=0.5, color='orange')
                    plt.title(f"Lamp spectrum - based on the trace of spec {number}")
                    plt.grid(linestyle=":",which="both")
                    plt.minorticks_on()
                    plt.ylabel("Counts")
                    plt.xlabel("Pixels")
                    plt.legend(loc='upper left')
                    for n in range(len(lamp_peak_positions)):
                        plt.text(lamp_peak_positions[n], lamp_peak_heights["peak_heights"][n],str(lamp_peak_positions[n]),rotation="vertical")

            if diagnostic_plots:
                plt.figure(figsize=(12,12*self.aspect_ratio)) 
                plt.imshow(np.log10(target_spec_data), vmax=np.log10(target_spec_data.max()), cmap = "gray", origin='lower')
                plt.plot(xvals, trace_center, 'C0', linewidth=1, label=f"Spine spec {number}")
                plt.fill_between(xvals, trace_center-trace_half_width, trace_center+trace_half_width, color='C1', alpha=0.4)
                plt.plot(xvals, trace_center_sky, color='C0', linestyle="--", linewidth=1, label="Sky spines")
                plt.fill_between(xvals, trace_center_sky-trace_half_width, trace_center_sky+trace_half_width, color='magenta', alpha=0.3)
                plt.plot(xvals, trace_center_sky2, color='C0', linestyle="--", linewidth=1)
                plt.fill_between(xvals, trace_center_sky2-trace_half_width, trace_center_sky2+trace_half_width, color='magenta', alpha=0.3)
                plt.title(f"Spectrum {number} - "+self.target_name+" field")
                plt.legend()
        
        if master_lamp_data is not None and diagnostic_plots:
            plt.figure(figsize=(12,12*self.aspect_ratio)) 
            plt.imshow(np.log10(lamp_image), vmax=np.log10(lamp_image.max()), cmap = "gray", origin='lower')
            for number,fitted_polymodel in enumerate(fitted_polymodel_list):
                lamp_trace = fitted_polymodel(xvals)
                plt.plot(xvals, lamp_trace, 'C0', linewidth=1, label=f"Spine spec {number}")
                plt.fill_between(xvals, lamp_trace-trace_half_width, lamp_trace+trace_half_width, color='C1', alpha=0.4)
            plt.title("Lamp spectrum")
            plt.legend()


        if spec_plots or diagnostic_plots:
            plt.show()
        
        if master_lamp_data is None:
            return spec_list
        else:           
            return spec_list, lamp_spec_list, lamp_peak_positions_list


    def wavelength_calibration(self, lamp_peak_positions_list, corresponding_wavelengths, unit = "angstrom", poly_order = 2, diagnostic_plots = True):

        """
        This function takes the list of lamp spectral peak positions (in pixels) and their corresponding wavelengths, and returns a list with
        the calibrated x-axis for one or more spectra.

        In case you need references for the lamp peaks, here are some databases you can use (as of 12 September, 2024):
         * TNG DOLORES spectrograph reference lamps can be found in table 1 here: https://www.tng.iac.es/instruments/lrs/
         * SOAR Goodman Spectrograph Reference Lamp Library: https://soardocs.readthedocs.io/projects/lamps/en/latest/
         * Observatorio do Pico dos Dias (OPD-LNA) spectrographs: https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd/instrumentacao/instrumentos-e-detectores

        Parameters
        ----------
        lamp_peak_positions_list: list
            This is one of the outputs from function extraction.extracting(). It contains the positions (in pixels) of all peaks in the lamp
            spectra with heights above the spectrum median. 
        corresponding_wavelengths: list
            List with wavelengths corresponding to each peak given in lamp_peak_positions_list. Even if the peak positions are slightly different
            for different traces, a single list with the corresponding wavelengths will do the job here.
        unit: string
            It can be any length unit accepted by astropy. Typically we use 'angstrom' or 'nanometer'.
        poly_order: integer
            The order of the polynomial used to find the wavelength solution.
        diagnostic_plots: boolean
            If True, easyspec will plot the wavelength solution and fit residuals.
        
        Returns
        -------
        wavelengths_list: list
            List with the wavelength solutions for each given spectrum.
        wavelengths_fit_std_list: list
            The standard deviation of the data points with respect to the wavelength solution for each given spectrum.
        wavelength_error_per_pixel_list: list
            The wavelength error per pixel for each given spectrum. This estimate is precise for a straight-line trace. For curved traces, it can
            be used as a rough estimate of the error.
        """

        try:
            _ = self.image_shape
        except:
            raise NameError("The variable self.image_shape does not exist. Please load your data with the function extraction.import_data() to avoid this issue.")


        wavelengths_list, wavelengths_fit_std_list, wavelength_error_per_pixel_list = [], [], []
        linfitter = LevMarLSQFitter()
        wlmodel = Polynomial1D(degree=poly_order)

        for number,lamp_peak_positions in enumerate(lamp_peak_positions_list):
            linfit_wlmodel = linfitter(model=wlmodel, x=lamp_peak_positions, y=corresponding_wavelengths)
            xrange = np.asarray(range(self.image_shape[1]))
            lcls = locals()
            exec( f"unit = u.{unit}", globals(), locals() )
            unit = lcls["unit"]
            wavelengths = linfit_wlmodel(xrange) * unit
            wavelengths_list.append(wavelengths)

            wavelength_error_per_pixel = np.sqrt(linfitter.fit_info['param_cov'][1][1])  # For polynomials with degree > 1, this error is only an approximation
            wavelength_error_per_pixel_list.append(wavelength_error_per_pixel)

            wavelengths_fit_std = np.std(linfit_wlmodel(lamp_peak_positions)-corresponding_wavelengths)
            wavelengths_fit_std_list.append(wavelengths_fit_std)
            
            if diagnostic_plots:
                print(f"Spectrum {number}:")
                print(f"Fit standard deviation = {wavelengths_fit_std} {str(unit)}")
                print("Wavelength error per pixel (linear approximation): ", wavelength_error_per_pixel, f"{unit}/pixel")

                plt.figure(figsize=(12,5))
                plt.plot(lamp_peak_positions, corresponding_wavelengths, 'o')
                plt.plot(xrange, wavelengths, '-',label=f"Wavelength solution\nError = {round(wavelength_error_per_pixel,5)} {unit}/pixel")
                plt.ylabel(f"$\lambda(x)$ [{str(unit)}]")
                plt.xlabel("x (pixels)")
                plt.title(f"Wavelength solution - spectrum {number}")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.legend()

                plt.figure(figsize=(12,5)) 
                plt.plot(lamp_peak_positions, linfit_wlmodel(lamp_peak_positions)-corresponding_wavelengths, 'o', label="Fit standard deviation = "+str(round(wavelengths_fit_std,3))+" "+str(unit))
                plt.plot(lamp_peak_positions,np.zeros(len(lamp_peak_positions)),'k--')
                plt.ylabel(f"Wavelength residuals [{str(unit)}]")
                plt.xlabel("x (pixels)")
                plt.grid(linestyle=":")
                plt.title(f"Fit residuals - spectrum {number}")
                plt.grid(linestyle=":",which="both")
                plt.minorticks_on()
                plt.legend()
            
        if diagnostic_plots:
            plt.show()
        

        return wavelengths_list, wavelengths_fit_std_list, wavelength_error_per_pixel_list
    
    def extinction_correction(self, spec_list, wavelengths_list, observatory, data_type, custom_observatory = None, spline_order = 1, plots = True):

        """
        This function removes the atmospheric extinction from the observed spectra.        

        Parameters
        ----------
        spec_list: list
            List with the extracted spectra of all the targets in the image or only the brightest spectrum. This is one of the outputs from function extraction.extracting().
        wavelengths_list: list
            List with the wavelength solutions for each given spectrum. This is the main result from function extraction.wavelength_calibration().
        observatory: string
            This is the observatory where you collected yoru data. Options are "ctio", "kpno", "lapalma", "mko", "mtham", "paranal", "apo".
        data_type: string
            Options are "target" or "std_star". We use this information to correctly select the airmass for the extinction correction.
        custom_observatory: string
            Optional. Path to a txt file containing the airmass extinction curve. This file must have two columns, the first with the wavelengths
            *in Angstroms* and the second with the extinction in mag/airmass.
        spline_order: integer
            The degree of the spline fit. Even values of spline_order should be avoided.
        plots: boolean
            If True, easyspec will plot a comparison between the corrected and original spectrum, as well as the extinction interpolation curve.
        
        Returns
        -------
        spec_atm_corrected_list: list
            List with all the spectra corrected for atmospheric extinction.
        """



        reference_dictionaty = {"ctio" : "The CTIO extinction curve was originally distributed with IRAF and comes from the work of Stone & Baldwin (1983 MN 204, 347) \
plus Baldwin & Stone (1984 MN 206, 241). The first of these papers lists the points from 3200-8370A while the second extended the \
flux calibration from 6056 to 10870A but the derived extinction curve was not given in the paper. The IRAF table follows SB83 out \
to 6436, the redder points presumably come from BS84 with averages used in the overlap region. More recent CTIO extinction curves \
are shown as Figures in Hamuy et al (92, PASP 104, 533 ; 94 PASP 106, 566).", 
"kpno" : "The KPNO extinction table was originally distributed with IRAF. The usage of this table is discouraged since the data provenance of this data is unclear.",
"lapalma" : "Extinction table for Roque de Los Muchachos Observatory, La Palma. Described in https://www.ing.iac.es/Astronomy/observing/manuals/ps/tech_notes/tn031.pdf",
"mko" : "Median atmospheric extinction data for Mauna Kea Observatory measured by the Nearby SuperNova Factory: https://www.aanda.org/articles/aa/pdf/2013/01/aa19834-12.pdf",
"mtham" : "Extinction table for Lick Observatory on Mt. Hamilton constructed from https://mthamilton.ucolick.org/techdocs/standards/lick_mean_extinct.html",
"paranal": "Updated extinction table for ESO-Paranal taken from https://www.aanda.org/articles/aa/pdf/2011/03/aa15537-10.pdf",
"apo" : "Extinction table for Apache Point Observatory. Based on the extinction table used for SDSS and available \
at https://www.apo.nmsu.edu/arc35m/Instruments/DIS/ (https://www.apo.nmsu.edu/arc35m/Instruments/DIS/images/apoextinct.dat)."}

        try:
            _ = self.airmass_target
        except:
            raise NameError("The variable self.airmass_target does not exist. Please load your data with the function extraction.import_data() to avoid this issue.")

        if data_type == "target":
            airmass = self.airmass_target
        elif data_type == "std_star":
            try:
                airmass = self.airmass_std_star
            except:
                raise RuntimeError("Airmass for the standard star was not found. Did you loaded the standard star data with the function extraction.tracing()?")
            if len(spec_list) > 1:
                raise RuntimeError(f"We have {len(spec_list)} spectra given as input. Since the variable data_type was set as 'std_star' by the user, the variable\
                                 'spec_list' must contain only one spectrum.")
        else:
            raise NameError("Invalid entry for data_type. The options are 'target' or 'std_star'.")

        if custom_observatory is None:
            airmass_extinction = np.loadtxt(str(libpath)+f"/{observatory}_airmass_extinction.txt")
            extinction_wavelength = airmass_extinction[:,0]
            extinction = airmass_extinction[:,1]
        else:
            airmass_extinction = np.loadtxt(custom_observatory)
            if np.shape(airmass_extinction)[1] != 2:
                raise RuntimeError("The custom_observatory file does not have two columns. Please be sure to use a txt file with two columns, the first with the wavelengths and the second with the extinction in mag/airmass.")
            extinction_wavelength = airmass_extinction[:,0]
            extinction = airmass_extinction[:,1]

        spec_atm_corrected_list = []
        for spec, wavelength in zip(spec_list,wavelengths_list):
            wavelength_min_index = self.find_nearest(extinction_wavelength, wavelength.value.min())
            wavelength_max_index = self.find_nearest(extinction_wavelength, wavelength.value.max())
            extinction_wavelength = extinction_wavelength[wavelength_min_index:wavelength_max_index]
            extinction = extinction[wavelength_min_index:wavelength_max_index]

            tck = interpolate.splrep(extinction_wavelength, extinction,k=spline_order)

            spec_atm_corrected = spec*2.51188643151**( airmass * interpolate.splev(wavelength.value, tck))  # The numerical factor here is 100**(1/5), which is the increase in flux when the magnitude decreases 1.
            spec_atm_corrected_list.append(spec_atm_corrected)

            if plots:
                plt.figure(figsize=(12,5)) 
                plt.plot(wavelength, spec, label="No atm correction")
                plt.plot(wavelength, spec_atm_corrected,color="orange",label="Atm corrected")
                plt.xlabel(f"Wavelength [${wavelength.unit}$]")
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                number = len(spec_atm_corrected_list)-1
                if data_type == "target":
                    plt.title(f"Atmospheric-corrected spectrum {number} - "+self.target_name+f" field - Airmass = {airmass}")
                else:
                    plt.title(f"Atmospheric-corrected spectrum for the standard star - Airmass = {airmass}")
                plt.legend()                    

                plt.figure(figsize=(12,5)) 
                plt.plot(wavelength.value, interpolate.splev(wavelength.value, tck),label="Fit Spline")
                if custom_observatory is None:
                    plt.plot(extinction_wavelength, extinction,label=f"Extinction data {observatory}")
                else:
                    plt.plot(extinction_wavelength, extinction,label="Extinction data custom observatory")
                plt.axhline(y=0.0, color='r', linestyle=':')
                plt.xlabel(f"Wavelength [${wavelength.unit}$]")
                plt.ylabel("Extinction (mag/airmass)")
                plt.ylim(-0.1,1.9)
                plt.minorticks_on()
                plt.grid(which="both", linestyle=":")
                if data_type == "target":
                    plt.title(f"Extinction interpolation for spectrum {number} - "+self.target_name+" field")
                else:
                    plt.title("Extinction interpolation for the standard star")
                plt.legend()

                plt.show()
        
        if plots and custom_observatory is None:
            print("Please cite the work where this extinction curve was measured:\n"+reference_dictionaty[observatory])

        return spec_atm_corrected_list

    def list_available_standards(self, std_star_dataset=None):

        """
        This function lists the available datasets and their standard stars.

        Parameters
        ----------
        std_star_dataset: string
            If None, the function will plot all available datasets. Options are "ctiocal", "irscal", "bstdscal", "spec16cal", "spechayescal", "iidscal", "spec50cal",
            "redcal", "ctionewcal", "ctio", "oke1990", "blackbody".

        Returns
        -------
        reference_dictionaty[dataset]: string
            If a dataset is given, the function will return the reference for this dataset.
        """

        reference_dictionaty = {"blackbody" :  "Blackbody flux distributions in various magnitude bands.",
"bstdscal" :  "The brighter KPNO IRS standards (i.e. those with HR numbers) at 29 bandpasses, data from various\
sources transformed to the Hayes and Latham system, unpublished.",
"ctiocal" :  "Fluxes for the southern tertiary standards as published by Baldwin & Stone, 1984, MNRAS,\
206, 241 and Stone and Baldwin, 1983, MNRAS, 204, 347.",
"ctionewcal" : "Fluxes at 50A steps in the blue range 3300-7550A for the tertiary standards of Baldwin and\
Stone derived from the revised calibration of Hamuy et al., 1992, PASP, 104, 533. This dataset also contains the fluxes of\
the tertiaries in the red (6050-10000A) at 50A steps as will be published in PASP (Hamuy et al 1994). The combined fluxes are\
obtained by gray shifting the blue fluxes to match the red fluxes in the overlap region of 6500A-7500A and averaging the red and\
blue fluxes in the overlap.  The separate red and blue fluxes may be selected by following the star name with 'red' or 'blue'; i.e. CD 32 blue.",
"iidscal" : "Dataset of the KPNO IIDS standards at 29 bandpasses, data from various sources transformed to the Hayes and Latham system, unpublished.",
"irscal" :  "Dataset of the KPNO IRS standards at 78 bandpasses, data from various sources transformed to the Hayes and Latham\
system, unpublished (note that in this dataset the brighter standards have no values - the `bstdscal' dataset must be used for these standards at this time).",
"oke1990" : "Dataset of spectrophotometric standards observed for use with the HST, Table VII, Oke 1990, AJ, 99. 1621 (no correction\
was applied). An arbitrary 1A bandpass is specified for these smoothed and interpolated flux 'points'.  Users may copy and modify these\
files for other bandpasses.",
"redcal" : "Dataset of standard stars with flux data beyond 8370A. These stars are from the IRS or the IIDS dataset but have data\
extending as far out into the red as the literature permits. Data from various sources.",
"spechayescal" : "The KPNO spectrophotometric standards at the Hayes flux points, Table IV, Spectrophotometric Standards, Massey et al., 1988, ApJ 328, p. 315.",
"spec16cal" : "Dataset containing fluxes at 16A steps in the blue range 3300-7550A for the secondary standards, published in Hamuy et al., 1992, PASP, 104, 533.\
This dataset also contains the fluxes of the secondaries in the red (6020-10300A) at 16A steps as will be published in PASP (Hamuy et al 1994).\
The combined fluxes are obtained by gray shifting the blue fluxes to match the red fluxes in the overlap region of 6500A-7500A and averaging the blue\
and red fluxes in the overlap.  The separate red and blue fluxes may be selected by following the star name with 'red' or 'blue'; i.e. HR 1544 blue.",
"spec50cal" : "The KPNO spectrophotometric standards at 50 A intervals. The data are from (1) Table V, Spectrophotometric Standards, Massey et al., 1988,\
ApJ 328, p. 315 and (2) Table 3, The Kitt Peak Spectrophotometric Standards: Extension to 1 micron, Massey and Gronwall, 1990, ApJ 358, p. 344."}

        available_datasets = glob.glob(str(libpath_std)+"/*")
        available_datasets = np.sort(available_datasets)

        if std_star_dataset is None:
            print("Available datasets:")
            for available_dataset in available_datasets:
                print(available_dataset.split("/")[-1])
        
        elif str(libpath_std)+"/"+std_star_dataset in available_datasets:
            available_std_stars = glob.glob(str(libpath_std)+f"/{std_star_dataset}/*")
            std_stars = []
            for available_std_star in available_std_stars:
                std_stars.append(available_std_star.split("/")[-1])

            return std_stars, reference_dictionaty[std_star_dataset]
        else:
            print("Input dataset not found in our library. Try using the function extraction.list_available_standards(dataset=None) to see the available datasets.")


    def std_star_normalization(self, spec_atm_corrected_std, wavelengths_std, std_star_dataset, std_star_archive_file, smooth_window = 101, exclude_regions = None, smooth_window_archive = 11, interpolation_order=1, plots = True):

        """
        This function normalizes the measured standard spectrum by its exposure time (read in the function extraction.import_data()) and
        by its archival spectrum.

        Parameters
        ----------
        spec_atm_corrected_std: list
            This is the standard star spectrum after being corrected for atmospheric extinction.
        wavelengths_std: list
            This is the wavelength solution for the standard star obtained with the function extraction.wavelength_calibration().
        std_star_dataset: string
            This is the dataset where you can find your standard star.The options are "ctiocal", "irscal", "bstdscal", "spec16cal",
            "spechayescal", "iidscal", "spec50cal", "redcal", "ctionewcal", "ctio", "oke1990", "blackbody".
        std_star_archive_file: string
            This is the name of the datafile of the archival standard star spectrum. You can find more by running the function
            extraction.list_available_standards(). In any case, it should be something like "l4364blue.dat".
        smooth_window: integer
            Must be an odd number. This is the number of neighbouring wavelength bins used to extract the standard star continuum
            with a median filter.
        exclude_regions: list
            List of regions (in Angstroms) to be excluded from the measured standard star spectrum when extracting its continuum, e.g.: exclude_regions = [[4000,5000],[8000,8500]].
        smooth_window_archive: integer
            Must be an odd number. Same as above but for the archival spectrum.   
        plots: boolean
            If True, easyspec will plot several diagnostic plots showing the step-by-step of the flux calibration solution.
        
        Returns
        -------
        correction_factor: array
            This is an array in astropy units of erg/cm2/s/Angstrom containing the flux calibration solution for the standard star.
        """

        if isinstance(spec_atm_corrected_std[0],np.ndarray):
            spec_atm_corrected_std = spec_atm_corrected_std[0]
        if isinstance(wavelengths_std[0], u.quantity.Quantity):
            wavelengths_std = wavelengths_std[0]

        if smooth_window%2 == 0:
            smooth_window = smooth_window - 1
            print(f"The input parameter 'smooth_window' must be odd. We are resetting it to {smooth_window}.")

        if smooth_window_archive%2 == 0:
            smooth_window_archive = smooth_window_archive - 1
            print(f"The input parameter 'smooth_window_archive' must be odd. We are resetting it to {smooth_window_archive}.")

        
        spec_atm_corrected_std = np.asarray(spec_atm_corrected_std)
        spec_atm_corrected_std = spec_atm_corrected_std/self.exposure_std_star  # Correcting for exposure
        smoothed_spec = medfilt(spec_atm_corrected_std, smooth_window)

        _, reference = self.list_available_standards(std_star_dataset=std_star_dataset)
        archival_data = np.loadtxt(libpath_std/Path(std_star_dataset)/Path(std_star_archive_file))
        archival_wavelength = archival_data[:,0] * u.angstrom
        archival_flux = archival_data[:,1] * u.ABmag
        archival_flux = archival_flux.to(u.erg / u.cm**2 / u.s / u.AA, equivalencies=u.spectral_density(archival_wavelength))
        archival_flux_smoothed = medfilt(archival_flux, smooth_window_archive)

        tck = interpolate.splrep(archival_wavelength.value, archival_flux_smoothed,k=interpolation_order)
        if exclude_regions is None:
            tck2 = interpolate.splrep(wavelengths_std.value, smoothed_spec,k=interpolation_order)
        else:
            wavelengths_to_exclude = np.ones([len(wavelengths_std.value)])
            if isinstance(exclude_regions[0],list):
                for wavelength_region in exclude_regions:
                    index = np.where((wavelengths_std.value > wavelength_region[0]) & (wavelengths_std.value < wavelength_region[1]))[0]
                    wavelengths_to_exclude[index] = 0
            elif isinstance(exclude_regions[0],float) or isinstance(exclude_regions[0],int):
                index = np.where((wavelengths_std.value > exclude_regions[0]) & (wavelengths_std.value < exclude_regions[1]))[0]
                wavelengths_to_exclude[index] = 0
            else:
                raise RuntimeError("Input value for exclude_regions is not correct. Please use one or more ranges of wavelength, like [3500,3700] or [[3800,3900],[5050,5180]].")
            wavelengths_to_exclude = wavelengths_to_exclude.astype(bool)
            tck2 = interpolate.splrep(wavelengths_std.value[wavelengths_to_exclude], smoothed_spec[wavelengths_to_exclude],k=interpolation_order)
        
        archival_model = interpolate.splev(wavelengths_std.value, tck)
        measured_spec_continuum = interpolate.splev(wavelengths_std.value, tck2)
        measured_spec_continuum[measured_spec_continuum <= 0.0] = archival_model[measured_spec_continuum <= 0.0]

        correction_factor = (archival_model/measured_spec_continuum)
        correction_factor = medfilt(correction_factor, 11) * u.erg / u.cm**2 / u.s / u.AA

        if plots:
            print(reference)
            plt.figure(figsize=(12,5)) 
            plt.plot(wavelengths_std, spec_atm_corrected_std, label="Measured std star spectrum")
            plt.plot(wavelengths_std, measured_spec_continuum, label = "Std star continuum")
            plt.xlabel(f"Wavelength [${wavelengths_std.unit}$]")
            plt.ylabel("Counts")
            plt.minorticks_on()
            plt.title("Standard star measured spectrum and continuum")
            plt.grid(which="both", linestyle=":")
            plt.legend()

            plt.figure(figsize=(12,5)) 
            plt.plot(archival_wavelength, archival_flux,label="Archival std star spectrum")
            plt.plot(archival_wavelength, archival_flux_smoothed,label="Archival std star continuum")
            plt.xlabel(f"Wavelength ({archival_wavelength.unit})")
            plt.ylabel(r"$F_{\lambda}$"+f"({archival_flux.unit})")
            plt.title("Standard star archival spectrum and continuum")
            plt.minorticks_on()
            plt.grid(which="both", linestyle=":")
            plt.legend()

            plt.figure(figsize=(12,5))
            plt.plot(wavelengths_std, correction_factor ,color="C0",label="Archival/measured")
            plt.xlabel(f"Wavelength ({wavelengths_std.unit})")
            plt.ylabel("Correction factor")
            plt.ylim(0,np.median(correction_factor.value)*10)
            plt.minorticks_on()
            plt.title("Flux correction curve")
            plt.grid(which="both", linestyle=":")
            plt.legend()

            plt.figure(figsize=(12,5)) 
            plt.plot(wavelengths_std, spec_atm_corrected_std*correction_factor,color="orange", alpha=0.8,label="Corrected-measured std star spec")
            plt.plot(archival_wavelength, archival_flux,color="C0", alpha=0.8,label="Archival std star spec")
            plt.xlabel(f"Wavelength ({wavelengths_std.unit})")
            plt.ylabel(r"$F_{\lambda}$"+f"({archival_flux.unit})")
            plt.title("Corrected standard star spectrum")
            plt.ylim(0,archival_flux.value.max()*1.2)
            plt.minorticks_on()
            plt.grid(which="both", linestyle=":")
            plt.legend()

            plt.show()

        return correction_factor
        

        

    def target_flux_calibration(self, wavelengths_list, spec_atm_corrected_list, correction_factor, reddening = None, Rv = None, wavelength_cuts = None, output_directory = "./", save_spec = True, plot = True):

        """
        This function normalizes the target spectra for exposure time, corrects them by reddening (optional) and calibrate them using
        the correction_factor obtained with the standard star.
        
        Parameters
        ----------
        wavelengths_list: list
            List with the wavelength solutions for all target spectra.
        spec_atm_corrected_list: list
            List with all the spectra corrected for atmospheric extinction.
        correction_factor: array
            This is the array (in astropy units of erg/cm2/s/Angstrom) containing the flux calibration solution for the standard star.
            This is the main result from the function extraction.std_star_normalization().
        reddening: float
            Optional. The E(B-V) reddening in magnitudes. This is the Galactic redening in the direction of your observation. You can look
            for the appropriate reddening for your target here: 'https://irsa.ipac.caltech.edu/applications/DUST/'.
        Rv: float
            visual extinction to reddening ratio. Default is 3.1, but depending on the case, even a value up to 5 can be used. You can
            check what is a reasonable value for your target here: 'https://irsa.ipac.caltech.edu/applications/DUST/'.
        wavelength_cuts: list
            List with two wavelength values. The spectrum within this wavelength range will survive the analysis. The rest will be removed.
            Example: wavelength_cuts = [3500,7500]. Assuming the units in *Angstroms*.
        output_directory: string
            A string with the path to the output directory.   
        save_spec: boolean
            If True, the spectra will be saved in the output directory as "spec_0.dat", "spec_1.dat" and so on.  
        plots: boolean
            If True, easyspec will plot several diagnostic plots showing the step-by-step of the flux calibration solution.
        
        Returns
        -------
        calibrated_flux_list: list
            This is a list with the spectra calibrated in flux.
        """

        if Rv is None:
            Rv = 3.1
        
        dust_extinction_model = F99(Rv=Rv)  # F99 is the Fitzpatrick (1999) Milky Way R(V) dependent model

        calibrated_flux_list = []
        for wavelengths,spec_atm_corrected in zip(wavelengths_list,spec_atm_corrected_list):
            if reddening is not None:
                calibrated_flux_list.append((spec_atm_corrected*correction_factor/self.exposure_target)/dust_extinction_model.extinguish(wavelengths,Ebv=reddening))
            else:
                calibrated_flux_list.append(spec_atm_corrected*correction_factor/self.exposure_target)

            if wavelength_cuts is None:
                wavelength_min_index = None
                wavelength_max_index = None
            else:
                wavelength_min_index = self.find_nearest(wavelengths, wavelength_cuts[0])
                wavelength_max_index = self.find_nearest(wavelengths, wavelength_cuts[1])

            wavelengths = wavelengths[wavelength_min_index:wavelength_max_index]
            calibrated_flux_list[-1] = calibrated_flux_list[-1][wavelength_min_index:wavelength_max_index]
            spec_number = len(calibrated_flux_list)-1

            if plot:
                plt.figure(figsize=(12,5))
                if reddening is not None:
                    plt.plot(wavelengths, calibrated_flux_list[-1], color='orange', label=f'Spec {spec_number}, E(B-V)={reddening}, R(V)={Rv}')
                else:
                    plt.plot(wavelengths, calibrated_flux_list[-1], color='orange', label=f'Spec {spec_number}, not corrected for reddening')
                plt.minorticks_on()
                plt.grid(which="both",linestyle=":")
                plt.xlim(wavelengths.value.min(),wavelengths.value.max())
                plt.ylim(0,calibrated_flux_list[-1].value.max()*1.2)
                plt.title(f"Calibrated spec {spec_number} - "+self.target_name+" field")
                plt.ylabel("F$_{\lambda}$ "+f"[{calibrated_flux_list[-1].unit}]",fontsize=12)
                plt.xlabel(f"Observed $\lambda$ [${wavelengths.unit}$]",fontsize=12)
                plt.legend()
                
        
            if save_spec:
                output_directory = Path(output_directory)
                np.savetxt(str(output_directory)+f"/{self.target_name}_spec_{spec_number}.dat", np.c_[wavelengths.value, calibrated_flux_list[-1].value], header="wavelength (Angstrom), Flux (erg/cm2/s/Angstrom)")

        if plot:
            plt.show()

        return calibrated_flux_list








            