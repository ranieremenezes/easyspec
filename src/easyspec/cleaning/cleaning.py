import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable #this is to adjust the colorbars
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from astropy.io import fits
from scipy import stats as st
from scipy import ndimage
from pathlib import Path
import warnings
from astropy.nddata import CCDData
from astropy import units as u
import ccdproc as ccdp
from . import alignment
from .. import data_io


class cleaning:

    """This class contains all the functions necessary to perform the image reduction/cleaning, including cosmic ray removal."""

    def __init__(self): 
        # Print the current version of easyspec       
        pass


    def data_paths(self,bias=None,flats=None,lamp=None,standard_star=None,targets=None,darks=None,data_file_extension="fits"):

        """
        This function collects the paths for all data files (bias, flats, lamp, standard star, and targets), as long as they have the extension ".fit*".
        
        IMPORTANT: please note that different kinds of data files must be in different directories. E.g., put all bias files in a directory called "bias", 
        all flat files in a directory called "flats" and so on.

        Parameters
        ----------
        bias,flats,lamp,standard_star,darks: strings
            Strings containing the path to the directory containing the data files.
        
        targets: string or list of strings
            You can pass a single directory path as a string or a list of paths. E.g. targets = "./spectra_target1" or targets = ["./spectra_target1","./spectra_target2"].

        data_file_extension: string
            Extension of the data files. Default = "fits"
        Returns
        -------
        all_raw_data: dict
            A dictionary containing the paths to all data files (dictionary keys) and the raw data for all images (dictionary values).
        
        """
        
        self.all_images_list = []  # Master list with the paths to all data files

        if bias is not None:
            if Path(bias).is_dir():
                # Make a list with the bias files:
                self.bias_list = np.sort(glob.glob(str(Path(bias).resolve())+f'/*.{data_file_extension}')).tolist()
                self.all_images_list = self.all_images_list + self.bias_list
            else:
                raise TypeError("Invalid directory for bias files.")
        else:
            self.bias_list = []

        if flats is not None:
            if Path(flats).is_dir():
                # Make a list with the flat images:
                self.flat_list = np.sort(glob.glob(str(Path(flats).resolve())+f'/*.{data_file_extension}')).tolist()
                self.all_images_list = self.all_images_list + self.flat_list
            else:
                raise TypeError("Invalid directory for flat files.")
        else:
            self.flat_list = []

        if lamp is not None:
            if Path(lamp).is_dir():
                # Make a list with the lamp images:
                self.lamp_list = np.sort(glob.glob(str(Path(lamp).resolve())+f'/*.{data_file_extension}')).tolist()
                self.all_images_list = self.all_images_list + self.lamp_list
            else:
                raise TypeError("Invalid directory for lamp files.")
        else:
            self.lamp_list = []

        if standard_star is not None:
            if Path(standard_star).is_dir():
                # Make a list with the standard star images:
                self.std_list = np.sort(glob.glob(str(Path(standard_star).resolve())+f'/*.{data_file_extension}')).tolist()
                self.all_images_list = self.all_images_list + self.std_list
            else:
                raise TypeError("Invalid directory for standard star files.")
        else:
            self.std_list = []


        if targets is not None:
            if isinstance(targets,str):
                if Path(targets).is_dir():
                    # Make a list with the raw science images:
                    self.target_list = np.sort(glob.glob(str(Path(targets).resolve())+f'/*.{data_file_extension}')).tolist()
                    self.all_images_list = self.all_images_list + self.target_list
                else:
                    raise TypeError("Invalid directory for target files.")
            elif isinstance(targets,list):
                self.target_list = []
                for target in targets:
                    if Path(target).is_dir():
                        # Make a list with the raw science images:
                        self.target_list = self.target_list + np.sort(glob.glob(str(Path(target).resolve())+f'/*.{data_file_extension}')).tolist()
                    else:
                        raise TypeError("Invalid directory for target files.")
                
                self.all_images_list = self.all_images_list + self.target_list
            else:
                raise RuntimeError("Variable 'target' must be a string or a list of strings.")
        else:
            self.target_list = []


        if darks is not None:
            if Path(darks).is_dir():
                # Make a list with the dark files:
                self.darks_list = np.sort(glob.glob(str(Path(darks).resolve())+f'/*.{data_file_extension}')).tolist()
                self.all_images_list = self.all_images_list + self.darks_list
            else:
                raise TypeError("Invalid directory for dark files.")
        else:
            self.darks_list = []

        # Here we create a dictionary with all raw images:
        all_raw_data = {}
        images_shape = []
        for image_name in self.all_images_list:
            raw_image_data = fits.getdata(image_name)
            all_raw_data[image_name] = raw_image_data
            images_shape.append(raw_image_data.shape)

        # Here we check if all the images have the same size:
        if len(set(images_shape)) > 1:
            print("Raw data files have different shapes. Here is a list of them:")
            for image_name in self.all_images_list:
                print("File: ",Path(image_name).name,". Shape: ",all_raw_data[image_name].shape)

            raise RuntimeError("One or more raw image files have different sizes. All data files must have the same dimesions. Please fix this before proceeding."
                               +" See the list above with the shapes of all data files.")
            
        return all_raw_data


    def trim(self,raw_image_data,x1,x2,y1,y2):

        """
        We use this function to trim the data
        
        Parameters
        ----------
        raw_image_data: dict
            Dictionary containing the path (keys) and the raw data (values) for one or several images.
        x1,x2,y1,y2: integers
            Values giving the cutting limits for the trim function.

        Returns
        -------
        raw_image_data_trimmed: dict
            A dictionary with all image data files trimmed
        """

        raw_image_data_trimmed = {}  # Here we create a new dictionary for the trimmed data.
        for image_data_path,raw_image in raw_image_data.items():
            raw_image_data_trimmed[image_data_path] = raw_image[y1:y2,x1:x2]  # Addind the trimmed data to the new dictionary.
        
        return raw_image_data_trimmed


    def plot_images(self,datacube,image_type="all", Ncols = 2, specific_image=None, vmin=None, vmax=None, vertical_scale=None):

        """
        Function to plot a matrix with all the images given in datacube
        
        Parameters
        ----------
        datacube: dict
            A dictionary containing the paths to all data files (dictionary keys) and the data for all images (dictionary values).
        image_type: string
            String with the type of file to plot. Options: "all", "bias", "flat", "lamp", "standard_star", "target", and "dark".
        Ncols: int
            Number of culumns for the grid.
        specific_image: int
            In case you want to see a single image from e.g. the bias or the target sample, use specific_image = number corresponding to your image. It goes
            from zero up to the total number (minus one) of files of a specific kind you have. For instance, if you want to display only the first bias file,
            use image_type = "bias", Ncols = 1, and specific_image = 0.
        vertical_scale: float
            This variable allows you to control the vertical scale between the plots.

        Returns
        -------
            A plot with 3 columns and several lines containing all the images passed in the variable datacube.
        
        """

        # Here we guarantee that Ncols is integer:
        Ncols = int(Ncols)
        if Ncols < 1:
            Ncols = 2
            warnings.warn("A temptative of setting Ncols < 1 was done. easyspec is setting it back to the default value.")

        if image_type == "all":
            imagenames = list(datacube.keys())
        elif image_type == "bias":
            imagenames = self.bias_list
        elif image_type == "flat":
            imagenames = self.flat_list
        elif image_type == "lamp":
            imagenames = self.lamp_list
        elif image_type == "standard_star":
            imagenames = self.std_list
        elif image_type == "target":
            imagenames = self.target_list
        elif image_type == "dark":
            imagenames = self.darks_list
        else:
            raise ValueError('Invalid image_type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')
        
        # Here we create an array containing the image data for all requested files:
        datacube = np.stack([datacube[image_frame] for image_frame in imagenames],axis=0)
        number_of_images = len(datacube)  # Number of individual images in the cube
        
        if specific_image is not None:
            datacube_single = []  
            datacube_single.append(datacube[specific_image])
            datacube = datacube_single
            imagenames = [imagenames[specific_image]]
            number_of_images = 1

        

        if number_of_images == 1:
            number_of_rows = 1
        elif number_of_images%Ncols > 0:
            number_of_rows = int(number_of_images/Ncols) + 1  # Number of image grid rows
        else:
            number_of_rows = int(number_of_images/Ncols)  # Number of image grid rows
        
        Matrix_shape = gridspec.GridSpec(number_of_rows, Ncols) # Define the image grid

        if vertical_scale is not None:
            plt.figure(figsize=(12,vertical_scale*number_of_rows)) # Set the figure size
        else:
            plt.figure(figsize=(12,3*number_of_rows)) # Set the figure size
        for i in range(number_of_images): 
            # In this loop we plot each individual image we have
            single_image_data = datacube[i]
            plt.subplot(Matrix_shape[i])
            single_image_name = Path(imagenames[i]).stem  # Here we take only the image file name, i.e., we strike out all the data path and the extension.
            plt.title(single_image_name)
            ax = plt.gca()
            im = ax.imshow(self._log_display(single_image_data), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            if specific_image is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.08)
                plt.colorbar(im, cax=cax,label="Counts in log scale")
            
            
        
        plt.show()
    
    def combined_header(self, file_list, method, Type, header_hdu_entry=0, exposure_header_entry=None,
                        airmass_header_entry=None, keep_varying=False):
 
        """
        Build the header of a master (combined) file from the headers of all input frames.
 
        The first frame's header is used as the starting point, but keywords whose values differ between the input
        frames are removed (they would describe only the first frame), except for the exposure time and airmass,
        which are replaced by values representative of the combined image, and DATE-OBS, which is set to the start of
        the earliest observation. Provenance keywords (NCOMBINE, COMBTYPE, IMCMBnnn) are also added.
 
        Parameters
        ----------
        file_list: list of strings
            Paths to the frames that were combined.
        method: string
            Combination method ("median", "mean" or "mode").
        Type: string
            Type of master ("bias", "flat", "lamp", "standard_star", "target", or "dark").
        header_hdu_entry: int
            HDU extension where the header is.
        exposure_header_entry: string
            Header keyword with the exposure time (e.g. "EXPTIME"). Optional. It is searched in all HDUs (see look_in_header).
        airmass_header_entry: string
            Header keyword with the airmass (e.g. "AIRMASS"). Optional. It is searched in all HDUs (see look_in_header).
        keep_varying: bool
            If True, keywords that differ between the input frames are kept with the first frame's value.
 
        Returns
        -------
        header: astropy.io.fits.Header
            The header for the master file.
        """
 
        headers = [fits.getheader(file, ext=header_hdu_entry) for file in file_list]
        header = headers[0].copy()
 
        # Structural keywords are rewritten by astropy when the file is saved. BZERO/BSCALE must be removed because
        # the master is stored as floating point (removing them is safer than setting BZERO = 0).
        for key in ("BZERO", "BSCALE", "BLANK"):
            header.remove(key, ignore_missing=True, remove_all=True)
        structural = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "COMMENT", "HISTORY", ""}
        special = {key for key in (exposure_header_entry, airmass_header_entry, "DATE-OBS") if key is not None}
 
        # Keywords whose value is not the same in all input frames:
        varying = []
        for key in set(header.keys()) - structural - special:
            values = [h.get(key) for h in headers]
            if any(value != values[0] for value in values[1:]):
                varying.append(key)
        if not keep_varying:
            for key in varying:
                header.remove(key, ignore_missing=True, remove_all=True)
 
        # Exposure time: a mean/median/mode combination has the count level of a SINGLE exposure, so the keyword
        # used later to convert counts into count rates must hold the representative single-frame exposure.
        if exposure_header_entry is not None:
            # look_in_header() searches all HDUs and raises an error if the keyword is missing in any frame.
            exposures = np.array([self.look_in_header(file, exposure_header_entry) for file in file_list], dtype=float)
            if np.ptp(exposures) > 1e-6 * np.max(exposures):
                warnings.warn(f"The frames combined into the master {Type} have different exposure times "
                              f"({exposures.min()} to {exposures.max()}). Consider normalizing them by their "
                              "exposure times before combining.")
            header[exposure_header_entry] = (float(np.mean(exposures)), "Mean exposure time of combined frames")
            header["EXPSUM"] = (float(np.sum(exposures)), "Summed exposure time of combined frames")
            header["MEDEXP"] = (float(np.median(exposures)), "Median exposure time of combined frames")
 
        if airmass_header_entry is not None:
            airmasses = np.array([self.look_in_header(file, airmass_header_entry) for file in file_list], dtype=float)
            header[airmass_header_entry] = (float(np.mean(airmasses)), "Mean airmass of combined frames")
            header["MDAIRMAS"] = (float(np.median(airmasses)), "Median airmass of combined frames")
 
        dates = sorted(h["DATE-OBS"] for h in headers if "DATE-OBS" in h)  # ISO 8601 strings sort chronologically
        if dates:
            header["DATE-OBS"] = (dates[0], "Start of the earliest combined frame")
            header["DATE-END"] = (dates[-1], "Start of the latest combined frame")
 
        # Provenance (IRAF-like convention):
        header["NCOMBINE"] = (len(file_list), "Number of combined frames")
        header["COMBTYPE"] = (method, "Combination method")
        for n, file in enumerate(file_list, start=1):
            if n > 999:
                break
            header[f"IMCMB{n:03d}"] = Path(file).name[:68]
        if varying and not keep_varying:
            header["HISTORY"] = "Keywords removed because they differ between combined frames: " + ", ".join(sorted(varying))
        header["HISTORY"] = f"Master {Type} generated with easyspec (method: {method})."
        return header

    def master(self,Type,trimmed_data,method="median",header_hdu_entry=0,exposure_header_entry=None,airmass_header_entry=None,plot=True,variance=None,mask=None):
 
        """
        Function to stack the data files into a master file.
 
        Parameters
        ----------
        Type: string
            Type of master to stack. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".
        trimmed_data: dict
            Dictionary containing trimmed data. At the end we select only the entries defined with the variable 'Type'.
        method: string
            Stacking method. Options are "median", "mean", or "mode".
        header_hdu_entry: int
            HDU extension where we can find the header. Please check your data before choosing this value.
        exposure_header_entry: string
            Header keyword for the exposure time (e.g. "EXPTIME"). Used only for Type = "target" or "standard_star". If given, this keyword
            is set to the mean exposure time of the combined frames (the count level of a mean/median master corresponds to a single
            exposure), and the keywords EXPSUM (summed), MEDEXP (median) and AVEXP (mean) are added.
        airmass_header_entry: string
            Header keyword for the airmass (e.g. "AIRMASS"). Used only for Type = "target" or "standard_star". If given, this keyword
            is set to the mean airmass of the combined frames, and the keywords MDAIRMAS (median) and AVAIRMAS (mean) are added.
        plot: bool
            Keep it as True if you want to plot the master bias/dark/flat/etc.
        variance: dict
            Optional, only for Type = "target" or "standard_star". Dictionary with the variance frames of the images being combined (e.g. the
            output of compute_variance()). If given, the variance of the master is propagated from them. See the Notes below.
        mask: dict
            Optional, only for Type = "target" or "standard_star". Dictionary with the data quality masks of the images being combined (e.g. the
            output of compute_variance()). The mask of the master flags a pixel if it is flagged in ALL frames (for "median" and "mode", which
            reject outliers) or in ANY frame (for "mean", and whenever only one or two
            frames are combined, since then the median equals the mean).
 
        Returns
        -------
        master: numpy array
            The master bias raw data.
        master_TYPE.fits: data file
            This function automatically saves a file named master_TYPE.fits in the working directory. If Type = "target" and there are
            several target directories, one master per target is saved (master_target_DIRECTORYNAME.fits) and a list of arrays is returned.
            The header of each master is built from the headers of the frames combined into it (see combined_header). If the variance of
            the master is known, it is saved in an extension called "VAR", and its data quality mask in an extension called "MASK" (see the
            module easyspec.data_io for the file format).
 
        Notes
        -----
        For Type = "flat", each frame is scaled to the median level of all flat frames before being combined, because the lamp intensity
        usually changes slightly between exposures (this is the standard procedure, e.g. IRAF's "flatcombine" with scale = "median").
 
        Variance of the master (in the same units as the data, squared):
 
        - For Type = "bias", "dark", "flat", and "lamp", the variance is estimated empirically, pixel by pixel, from the scatter between the
          combined frames, and divided by the number of frames (times ~pi/2 for the median, whose exact factor for small numbers of frames is
          computed numerically). Differences in the overall level of the frames (e.g. lamp intensity drifts between flats, bias level drifts)
          are removed before computing the scatter, so that only the pixel noise is measured. At least two frames are needed.
          IMPORTANT: this variance does NOT include the noise of other masters previously subtracted from these frames (e.g. the master bias
          subtracted from the darks). This is intentional: compute_variance() combines all terms, taking into account that the same master
          bias is subtracted from the darks and from the science frames.
        - For Type = "target" and "standard_star", the variance is propagated from the input 'variance' dictionary, if given. The scatter
          between science frames is not used, because it is dominated by real changes (seeing, sky level, residual misalignments).
          The noise of the calibration masters (bias, dark, flat) is the same in all frames, so it does not average down: it is combined
          separately from the independent noise (photon and read noise), using the attribute 'self.calibration_variance' created by
          compute_variance().
 
        The masters, their variances and their masks are stored in the attributes 'self.masters', 'self.master_variances' and
        'self.master_masks' (dictionaries whose keys are "bias", "dark", "flat", "lamp", "standard_star" or "target_DIRECTORYNAME").
        compute_variance() and norm_master_flat() use them automatically.
        """
 
        if Type == "bias":
            data_list = self.bias_list
        elif Type == "flat":
            data_list = self.flat_list
        elif Type == "lamp":
            data_list = self.lamp_list
        elif Type == "standard_star":
            data_list = self.std_list
        elif Type == "target":
            data_list = self.target_list
        elif Type == "dark":
            data_list = self.darks_list
        else:
            raise ValueError('Invalid Type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')
 
        if method not in ["median", "mean", "mode"]:
            raise ValueError(f"Invalid method '{method}'. Options are: 'median', 'mean', and 'mode'.")
 
        if variance is not None:
            if Type not in ("target", "standard_star"):
                raise ValueError("The parameter 'variance' can only be used with Type = 'target' or 'standard_star'. The variance of calibration "
                                 "masters is estimated automatically from the scatter between the frames.")
            missing = [Path(frame).name for frame in data_list if frame not in variance]
            if missing:
                raise ValueError("The 'variance' dictionary has no entry for: " + ", ".join(missing) + ".")
        if mask is not None:
            if Type not in ("target", "standard_star"):
                raise ValueError("The parameter 'mask' can only be used with Type = 'target' or 'standard_star'.")
            missing = [Path(frame).name for frame in data_list if frame not in mask]
            if missing:
                raise ValueError("The 'mask' dictionary has no entry for: " + ", ".join(missing) + ".")
        for name, dictionary in (("variance", variance), ("mask", mask)):
            if dictionary is not None:
                wrong_shape = [Path(frame).name for frame in data_list if np.shape(dictionary[frame]) != np.shape(trimmed_data[frame])]
                if wrong_shape:
                    raise ValueError(f"The {name} of the following images has a different shape from the images themselves: " + ", ".join(wrong_shape)
                                     + f". The {name} must undergo the same geometric operations as the data (e.g. align()).")
 
        data_cube = np.stack([trimmed_data[frame] for frame in data_list],axis=0)
 
        # If Type == target, we have to check if we have more than one target:
        if Type == "target":
            data_path = ""
            data_splitter = []
            for n,file in enumerate(data_list):
                parent_directory = str(Path(file).parent.resolve())
                if parent_directory != data_path:
                    data_path = parent_directory
                    data_splitter.append(n)
            data_splitter.append(len(data_list))
        else:
            data_splitter = [0,len(data_list)]
 
 
        if method == "mode":
            if str(data_cube[0].dtype).find("int") < 0:
                method = "median"
                warnings.warn("The 'mode' method only works if the pixels have integer values. The pixels in the given file are of the "+str(data_cube[0].dtype)+
                              " type. easyspec will reset method='median'.")
 
        if not hasattr(self, "masters"):
            self.masters = {}
            self.master_variances = {}
        if not hasattr(self, "master_masks"):
            self.master_masks = {}
 
        master_list = []
        for n in range(len(data_splitter[:-1])):
 
            group = data_cube[data_splitter[n]:data_splitter[n+1]]
            n_frames = group.shape[0]
 
            # Flats: the lamp intensity changes slightly from exposure to exposure. Each flat is scaled to the median level of the group
            # before combining, otherwise the median would be dominated by these level differences (and not reject outliers properly).
            if Type == "flat" and n_frames > 1 and method != "mode":  # Scaling would make the pixel values non-integer, breaking "mode"
                levels = np.median(group.reshape(n_frames, -1), axis=1).astype(float)
                if np.all(levels > 0):
                    group = group * (np.median(levels) / levels)[:, None, None]
                else:
                    warnings.warn("Some flat frames have a non-positive median level, so the flats were combined without scaling.")
 
            if method == "median":
                master = np.median(group, axis=0)  # To combine with a median
            elif method == "mode":
                master = st.mode(group, axis=0, keepdims=True)[0][0].astype(int)  # To combine with a mode
            elif method == "mean":
                master = np.mean(group, axis=0)  # To combine with a mean
 
            if Type == "target":
                target_name = Path(data_list[data_splitter[n]]).parent.resolve().name
                key = f"target_{target_name}"
            else:
                key = Type
 
            # Variance of the master:
            if variance is not None:
                group_frames = data_list[data_splitter[n]:data_splitter[n+1]]
                factor = 1.0 if method == "mean" else self._median_variance_factor(n_frames)
                # Part of the variance due to the calibration masters, which is the same random error in all frames (see compute_variance()):
                calibration = getattr(self, "calibration_variance", {})
                common = [calibration.get(frame) for frame in group_frames]
                if n_frames == 1:
                    master_variance = np.asarray(variance[group_frames[0]], dtype=float)  # a single frame: the variance is passed unchanged
                elif all(c is not None and np.shape(c) == np.shape(variance[frame]) for c, frame in zip(common, group_frames)):
                    common = [np.asarray(c, dtype=float) for c in common]
                    independent = sum(np.clip(np.asarray(variance[frame], dtype=float) - c, 0, None)
                                      for frame, c in zip(group_frames, common)) / n_frames**2
                    # Fully correlated errors add linearly in standard deviation (and pass unchanged through a median):
                    common_part = (sum(np.sqrt(c) for c in common) / n_frames)**2
                    master_variance = factor * independent + common_part
                else:
                    warnings.warn("The calibration part of the variance (see compute_variance()) is not available for all frames, so the "
                                  "noise of the calibration masters is treated as independent between frames. The variance of the master "
                                  f"{Type} is therefore underestimated. Please compute the variance with compute_variance() in this session "
                                  "(and pass it through align(), if you align the images).")
                    master_variance = factor * sum(np.asarray(variance[frame], dtype=float) for frame in group_frames) / n_frames**2
                variance_method = "propagated"
            elif Type in ("bias", "dark", "flat", "lamp"):
                master_variance = self._empirical_master_variance(group, method, multiplicative=(Type == "flat"), label=f"master {Type}")
                variance_method = "empirical"
            else:
                master_variance = None
                variance_method = None
 
            # Data quality mask of the master: median/mode reject outliers, so a pixel is bad only if it is bad in all frames;
            # the mean is affected by any bad value, so a pixel is bad if it is bad in any frame.
            if mask is not None:
                group_masks = np.stack([np.asarray(mask[frame], dtype=np.uint8) for frame in data_list[data_splitter[n]:data_splitter[n+1]]])
                # (with 1 or 2 frames the median equals the mean, so the "mean" rule applies)
                averaging = method == "mean" or n_frames <= 2
                master_mask = np.bitwise_or.reduce(group_masks, axis=0) if averaging else np.bitwise_and.reduce(group_masks, axis=0)
            else:
                master_mask = None
 
            self.masters[key] = master
            self.master_variances[key] = master_variance
            self.master_masks[key] = master_mask
 
            # Saving master file. The header is built from the headers of the frames combined in THIS master
            # (see combined_header): keywords that vary between frames are removed, and exposure time, airmass
            # and DATE-OBS are replaced by values representative of the combined image.
            if Type == "target" or Type == "standard_star":
                allow_exp_and_airmass = True
            else:
                allow_exp_and_airmass = False
 
            hdr = self.combined_header(data_list[data_splitter[n]:data_splitter[n+1]], method, Type,
                                       header_hdu_entry=header_hdu_entry,
                                       exposure_header_entry=exposure_header_entry if allow_exp_and_airmass else None,
                                       airmass_header_entry=airmass_header_entry if allow_exp_and_airmass else None)
 
            # Keywords kept for backward compatibility with previous easyspec versions:
            if allow_exp_and_airmass and exposure_header_entry is not None and exposure_header_entry in hdr:
                hdr['AVEXP'] = (hdr[exposure_header_entry], 'Average exposure time for all observations')
            if allow_exp_and_airmass and airmass_header_entry is not None and airmass_header_entry in hdr:
                hdr['AVAIRMAS'] = (hdr[airmass_header_entry], 'Average airmass')
 
            if variance_method is not None:
                hdr['VARMETH'] = (variance_method, 'How the variance (VAR extension) was obtained')
                if variance_method == "empirical":
                    hdr['COMMENT'] = 'The variance (VAR extension) does not include the noise of masters previously subtracted from the input frames.'
            in_electrons = all(frame in getattr(self, "gain_used", {}) for frame in data_list[data_splitter[n]:data_splitter[n+1]])
            if Type == "target":
                output_file = 'master_'+Type+f'_{target_name}.fits'
            else:
                output_file = 'master_'+Type+'.fits'
            data_io.write_reduced_frame(output_file, master, header=hdr, variance=master_variance, mask=master_mask,
                                        units="electron" if in_electrons else None)
 
            if len(data_splitter) > 2:
                master_list.append(master)
 
            if plot:
                image_shape = np.shape(master)
                aspect_ratio = image_shape[0]/image_shape[1]
                plt.figure(figsize=(12,12*aspect_ratio))
                if Type == "target":
                    plt.title('Master '+Type+f' {target_name} - Method: '+method)
                else:
                    plt.title('Master '+Type+' - Method: '+method)
                ax = plt.gca()
                if Type == "bias":
                    vmax = np.median(self._log_display(master))
                    im = ax.imshow(self._log_display(master), origin='lower', cmap='gray', vmax=vmax)
                else:
                    im = ax.imshow(self._log_display(master), origin='lower', cmap='gray')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.08)
                plt.colorbar(im, cax=cax,label="Counts in log scale")
 
            if n == (len(data_splitter[:-1]) - 1):
                plt.show()
 
        if len(data_splitter) > 2:
            return master_list
        else:
            return master

    @staticmethod
    def _median_variance_factor(n_frames):
 
        """
        Ratio between the variance of the median and the variance of the mean of n_frames independent Gaussian values with the same variance.
        It tends to pi/2 for large n_frames; for small n_frames it is computed numerically (e.g. ~1.35 for 3 frames). For n_frames <= 2 the
        median equals the mean, so the ratio is 1.
        """
 
        if n_frames <= 2:
            return 1.0
        rng = np.random.default_rng(20261015)  # Fixed seed: the result is deterministic
        draws = rng.standard_normal((max(20000, 4_000_000 // n_frames), n_frames))
        return float(np.var(np.median(draws, axis=1)) * n_frames)
 
 
    def _empirical_master_variance(self, group, method, multiplicative, label):
 
        """
        Pixel-by-pixel variance of a master combined from the frames in 'group' (array with shape [n_frames, ny, nx]), estimated from the
        scatter between the frames. Differences in the overall level of the frames are removed first (multiplicatively if multiplicative=True,
        e.g. for flats; additively otherwise), so that only the pixel noise is measured. Returns None if there is only one frame.
        """
 
        n_frames = group.shape[0]
        if n_frames < 2:
            warnings.warn(f"The {label} was built from a single frame, so its variance cannot be estimated from the data. "
                          "It will be treated as noiseless, which underestimates the final uncertainties.")
            return None
 
        levels = np.median(group.reshape(n_frames, -1), axis=1).astype(float)
        reference_level = np.median(levels)
        first = None
        sum_1 = np.zeros(group.shape[1:])
        sum_2 = np.zeros(group.shape[1:])
        for i in range(n_frames):
            frame = group[i].astype(float)
            if multiplicative and levels[i] > 0:
                frame = frame * (reference_level / levels[i])
            elif not multiplicative:
                frame = frame - levels[i] + reference_level
            if first is None:
                first = frame
            difference = frame - first  # Shifted data: avoids numerical cancellation when the level is much larger than the noise
            sum_1 += difference
            sum_2 += difference**2
        single_frame_variance = np.clip((sum_2 - sum_1**2 / n_frames) / (n_frames - 1), 0, None)
 
        factor = 1.0 if method == "mean" else self._median_variance_factor(n_frames)
        return factor * single_frame_variance / n_frames

    def debias(self, trimmed_data, masterbias, Type="all", pad_with_zeros=False):

        """
        Function to subtract the masterbias from the rest of the data.
        
        Parameters
        ----------
        trimmed_data: dict
            Dictionary containing trimmed data. It can even contain the raw bias files. At the end we select only the entry given in the variable Type.
        masterbias: array
            Array with master bias raw data.
        Type: string
            Type of data files to subtract the master bias. Options are: "all", "flat", "lamp", "standard_star", "target", and "dark".
        pad_with_zeros: bool
            If True, easyspec will look for negative-value pixels after the bias subtraction and set them to zero. 

        Returns
        -------
        debias_data_out: dict
            Dictionary containing the debiased data.
        """

        valid_types = ["all", "flat", "lamp", "standard_star", "target", "dark"]
        if Type not in valid_types:
            raise ValueError(f"Invalid Type '{Type}'. Options are: " + ", ".join(f'"{t}"' for t in valid_types) + ".")

        debias_list = []
        for file_name in trimmed_data.keys():
            if Type == "all" and file_name not in self.bias_list:
                debias_list = debias_list + [file_name]
            elif Type == "flat" and file_name in self.flat_list:
                debias_list = debias_list + [file_name]
            elif Type == "lamp" and file_name in self.lamp_list:
                debias_list = debias_list + [file_name]
            elif Type == "standard_star" and file_name in self.std_list:
                debias_list = debias_list + [file_name]
            elif Type == "target" and file_name in self.target_list:
                debias_list = debias_list + [file_name]
            elif Type == "dark" and file_name in self.darks_list:
                debias_list = debias_list + [file_name]

        debias_data_out = {}  # Dictionary for the debiased images

        for i in range(len(debias_list)):  
            debias_data_out[debias_list[i]] = trimmed_data[debias_list[i]] - masterbias  # Subtracting the master bias from each one of the raw images
            a,b=np.where(debias_data_out[debias_list[i]]<=0)
            if len(a) > 0 and pad_with_zeros:
                debias_data_out[debias_list[i]][a,b] = 0  # Padding negative pixels with zeros
        
        return debias_data_out


    def sub_dark(self, debiased_data, masterdark, Type="all", pad_with_zeros=False, exposure_header_entry=None, dark_exposure=None):

        """
        Function to subtract the masterdark from the rest of the data.

        IMPORTANT: the master dark must be bias-subtracted, i.e., it must contain only the dark current. This is the case if you ran debias() on the
        dark files (debias with Type="all" already does it) before creating the master dark with master("dark", ...).

        Parameters
        ----------
        debiased_data: dict
            Dictionary containing debiased data. This dictionary can even contain the raw dark files, but should not contain the raw bias files.
            At the end we select only the entry given in the variable Type.
        masterdark: array
            Array with master dark raw data (bias-subtracted).
        Type: string
            Type of data files to subtract the master dark. Options are: "all", "flat", "lamp", "standard_star", and "target".
        pad_with_zeros: bool
            If True, easyspec will look for negative-value pixels after the dark subtraction and set them to zero.
        exposure_header_entry: string
            Header keyword with the exposure time (e.g. "EXPTIME"). If given, the master dark is scaled by t_image/t_dark before being subtracted from
            each image, where t_image is read from the header of each image. This is necessary whenever the darks and the other frames have different
            exposure times. If None (default), the master dark is subtracted without scaling, which is only correct if all frames have the same exposure
            time as the darks.
        dark_exposure: float
            Exposure time of the master dark, in the same units as the images' exposure times. Used only if exposure_header_entry is given. If None,
            easyspec reads it from the headers of the dark files listed by data_paths().

        Returns
        -------
        subdark_data_out: dict
            Dictionary containing the data corrected for dark.

        Notes
        -----
        The scaling factor applied to each image is stored in the attribute 'self.dark_scale_factors'. Scaling assumes that the dark current grows
        linearly with time. This is a good approximation for most pixels, but hot pixels may deviate from it, so the best results are obtained when the
        darks have the same (or a longer) exposure time as the frames being corrected.
        """

        valid_types = ["all", "flat", "lamp", "standard_star", "target"]
        if Type not in valid_types:
            raise ValueError(
                'Invalid Type. Options are: "all", "flat", "lamp", "standard_star", and "target".'
            )

        # Exposure time of the master dark (only needed if we are going to scale it):
        if exposure_header_entry is not None and dark_exposure is None:
            if len(self.darks_list) == 0:
                raise ValueError("No dark files were given in data_paths(), so easyspec cannot read the dark exposure time. "
                                 "Please provide it with the parameter 'dark_exposure'.")
            dark_exposures = np.array([self.look_in_header(file, exposure_header_entry) for file in self.darks_list], dtype=float)
            if np.ptp(dark_exposures) > 1e-6 * np.max(dark_exposures):
                warnings.warn(f"The dark files have different exposure times ({dark_exposures.min()} to {dark_exposures.max()}). "
                              f"easyspec will assume that the master dark has the mean exposure time ({dark_exposures.mean()}).")
            dark_exposure = float(np.mean(dark_exposures))
        if exposure_header_entry is not None and dark_exposure <= 0:
            raise ValueError(f"Invalid dark exposure time: {dark_exposure}. It must be positive.")

        subdark_list = []
        for file_name in debiased_data.keys():
            if Type == "all" and file_name not in self.darks_list:
                subdark_list = subdark_list + [file_name]
            elif Type == "flat" and file_name in self.flat_list:
                subdark_list = subdark_list + [file_name]
            elif Type == "lamp" and file_name in self.lamp_list:
                subdark_list = subdark_list + [file_name]
            elif Type == "standard_star" and file_name in self.std_list:
                subdark_list = subdark_list + [file_name]
            elif Type == "target" and file_name in self.target_list:
                subdark_list = subdark_list + [file_name]


        subdark_data_out = {}  # Dictionary for the sub_dark images
        self.dark_scale_factors = {}

        for i in range(len(subdark_list)):
            if exposure_header_entry is not None:
                image_exposure = float(self.look_in_header(subdark_list[i], exposure_header_entry))
                scale = image_exposure / dark_exposure  # The dark current is proportional to the exposure time
            else:
                scale = 1.0
            self.dark_scale_factors[subdark_list[i]] = scale

            subdark_data_out[subdark_list[i]] = debiased_data[subdark_list[i]] - scale * masterdark  # Subtracting the (scaled) master dark from each one of the images
            a,b=np.where(subdark_data_out[subdark_list[i]]<=0)
            if len(a) > 0 and pad_with_zeros:
                subdark_data_out[subdark_list[i]][a,b] = 0  # Padding negative pixels with zeros

        return subdark_data_out


    def norm_master_flat(self, masterflat, method="polynomial", degree=5, smooth_window=51, surface_degree_y=5, surface_degree_x=5, illumination_threshold=0.5,
                         low_signal_threshold=0.1, clip_sigma=3.0, max_iterations=10, bin_size=8, header_hdu_entry=0, plots=True,
                         output_file="norm_master_flat.fits", masterflat_variance=None, bad_column_window=15, surface_profile="median_filter", max_flat_noise=None):

        """
        Function to normalize the master flat.

        A spectroscopic flat field contains three ingredients: (1) the pixel-to-pixel sensitivity of the detector, which is what we want to correct;
        (2) the spectrum of the flat-field lamp multiplied by the instrumental response along the dispersion axis, which must be removed, otherwise
        it would be imprinted on all science spectra; and (3) the illumination profile along the slit (spatial axis). The normalization divides the
        master flat by a smooth model of (2), or of (2) and (3), depending on the method:

        - "polynomial" (default) and "median_filter": a 1D model along the dispersion axis (x) is built from the illuminated rows and divided out of
          every row. The normalized flat keeps the pixel-to-pixel sensitivity AND the relative illumination along the slit (e.g. slit-width
          irregularities or dust on the slit, which also affect the science frames). This is the standard approach (e.g. IRAF's "response" task).
        - "surface": a two-step model, as in IRAF's "response" + "illumination" tasks. First, the 1D model along the dispersion axis is built
          exactly as in the "polynomial" or in the "median_filter" method (see 'surface_profile'). Then the ratio masterflat/(1D model), which is ~1 and contains only the
          illumination along the slit and the vignetting, is fitted with a smooth 2D Legendre polynomial of degree 'surface_degree_x' in x and
          'surface_degree_y' in y (Poisson-weighted, with sigma clipping). The model is the product of the two. The normalized flat keeps only the
          pixel-to-pixel (small-scale) sensitivity: the large-scale illumination along the slit is removed. Use it when the lamp illumination
          along the slit is not representative of the sky (e.g. you will correct the slit illumination with twilight flats), or when the response
          varies smoothly along the slit (e.g. vignetting). Small-scale slit features (dust) are partially kept.

        In all cases the dispersion axis must be horizontal. The 1D profile is computed with a median along the illuminated rows, after dividing each
        row by its own level, and its uncertainty is estimated column by column. Bad columns (narrow outliers of the profile) are excluded from
        the models, so that they remain in (and are corrected by) the normalized flat.

        IMPORTANT: the master flat must be bias- (and dark-, if relevant) subtracted before calling this function.

        Parameters
        ----------
        masterflat: array
            Array with master flat raw data.
        method: string
            Method to be adopted to model the master flat large-scale shape (see plots for clarification). Options are "polynomial",
            "median_filter", and "surface".
        degree: int
            Polynomial degree of the 1D model along the dispersion axis. Used if method = "polynomial" or "surface".
        smooth_window: integer
            Must be an odd number. This is the number of neighbouring pixels used to extract the flat field profile with a median filter. Used only if
            method = "median_filter".
        surface_degree_y: int
            Degree of the illumination surface along the spatial axis. Used only if method = "surface".
        surface_degree_x: int
            Degree of the illumination surface along the dispersion axis. It describes only slow changes of the slit illumination with
            wavelength (e.g. vignetting), since the lamp spectrum is already removed by the 1D model. Used only if method = "surface". Default = 2.
        illumination_threshold: float
            Rows whose median level is below illumination_threshold times the level of the brightest rows are considered not illuminated by the slit
            and are not used to build the model. Default = 0.5.
        low_signal_threshold: float
            Pixels where masterflat/model < low_signal_threshold (e.g. outside the slit, vignetted regions, dead pixels) or > 1/low_signal_threshold
            (places where the model is unreliable) are set to 1 in the normalized flat, i.e., they are left uncorrected, avoiding divisions by ~0
            when flattening the data. Columns whose profile is below low_signal_threshold times the median profile are not used in the fits and
            are also left uncorrected. Default = 0.1.
        clip_sigma: float
            Threshold, in units of the robust scatter, used to identify bad columns and in the iterative sigma clipping of the surface fit.
        max_iterations: int
            Maximum number of sigma-clipping iterations of the surface fit.
        bin_size: int
            Size (in pixels) of the square bins used to build the "surface" model. Used only if method = "surface".
        header_hdu_entry: int
            HDU extension where we can find the header. Please check your data before choosing this value.
        plots: bool
            If True, easyspec will plot the master flat profile, fit residuals and normalized master flat.
        output_file: string
            Name (or path) of the fits file where the normalized master flat is saved. Default = "norm_master_flat.fits".
        masterflat_variance: array
            Variance of the master flat. If None and masterflat is the array returned by master("flat", ...) in this session, easyspec uses
            the variance computed by master(). If you loaded the master flat from a file, you can get its variance with read_master().
        surface_profile: string
            Method used to build the 1D model (step 1) of the "surface" method: "polynomial" (degree 'degree') or "median_filter"
            (default, window 'smooth_window'). Use "median_filter" when the profile has structures that a polynomial cannot follow (a large reduced
            chi-square and oscillating residuals in the first diagnostic plot).
        max_flat_noise: float
            Optional. Maximum relative noise per pixel of the master flat (e.g. 0.01 for 1%). Columns where the median relative noise of the
            master flat, sqrt(Var)/masterflat (smoothed along x over 'bad_column_window' pixels), is larger than this value are left uncorrected
            (set to 1). Flat-fielding divides the data by a noisy estimate of the pixel sensitivity: where the noise of the flat is larger than
            the pixel-to-pixel sensitivity variations it corrects (typically ~0.3-1%), applying it adds more noise than it removes. This
            typically happens at the faint (often blue) end of the lamp spectrum. Requires the variance of the master flat. Default = 0.01
            (you can also set it to None). The noise of each column is printed and shown in the first diagnostic plot, to help choosing this value.
        bad_column_window: int
            Width (in pixels) of the window used to identify bad columns: each column is compared with the medians of its bad_column_window//2
            neighbours on each side, and flagged if it deviates by more than clip_sigma times the robust scatter. Bad columns are excluded from
            the models and therefore corrected by the flat. Structures broader than ~bad_column_window/2 are treated as real features of the
            profile. Default = 15.

        Returns
        -------
        normalized_master_flat: array
            Normalized master flat raw data.
        norm_master_flat.fits: data file
            This function automatically saves a file named "norm_master_flat.fits" (see output_file) in the working directory.

        Notes
        -----
        Information about the fit (method, degrees, reduced chi-square of the 1D profile fit, number of rejected points, illuminated rows) is stored
        in the attribute 'self.flat_fit_info'. The boolean mask of the pixels set to 1 because of low signal is stored in 'self.flat_low_signal_mask'.
        The variance of the normalized flat, Var(masterflat)/model^2 (the uncertainty of the smooth model itself is neglected, since it is
        constrained by many pixels), is stored in 'self.norm_flat_variance' and saved in the "VAR" extension of the output file. It is zero for
        the pixels set to 1. The normalized flat is stored in 'self.norm_flat', so that flatten() can recognize it and use its variance.
        """

        # ---------------------- Input validation ----------------------
        valid_methods = ["polynomial", "median_filter", "surface"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Options are: " + ", ".join(f'"{m}"' for m in valid_methods) + ".")
        if max_flat_noise is not None and not max_flat_noise > 0:
            raise ValueError("'max_flat_noise' must be a positive number (e.g. 0.006 for 0.6%) or None.")
        if surface_profile not in ("polynomial", "median_filter"):
            raise ValueError(f"Invalid surface_profile '{surface_profile}'. Options are: \"polynomial\" and \"median_filter\".")
        # Method used to build the 1D model along the dispersion axis:
        profile_method = surface_profile if method == "surface" else method
        for name, value in (("degree", degree), ("surface_degree_y", surface_degree_y), ("surface_degree_x", surface_degree_x),
                            ("bad_column_window", bad_column_window)):
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"'{name}' must be a non-negative integer.")
        if smooth_window%2 == 0:
            smooth_window = smooth_window - 1
            print(f"The input parameter 'smooth_window' must be odd. We are resetting it to {smooth_window}.")

        # If the master flat comes from master() in this session, we can recover its variance:
        if masterflat_variance is None and masterflat is getattr(self, "masters", {}).get("flat"):
            masterflat_variance = self.master_variances.get("flat")

        masterflat = np.asarray(masterflat, dtype=float)
        ny, nx = masterflat.shape
        xvals = np.arange(nx)

        # ---------------------- Illuminated rows and 1D profile ----------------------
        row_levels = np.nanmedian(masterflat, axis=1)
        reference_level = np.nanpercentile(row_levels, 95)
        if not reference_level > 0:
            raise ValueError("The master flat has no positive signal. Please check that it was correctly bias-subtracted.")
        illuminated = row_levels > illumination_threshold*reference_level
        n_illuminated = int(np.count_nonzero(illuminated))
        if n_illuminated < 3:
            raise ValueError("Less than 3 rows of the master flat are illuminated. Please check the data or lower 'illumination_threshold'.")

        # Each illuminated row is divided by its own level, so that the slit illumination does not inflate the scatter between rows:
        relative_rows = masterflat[illuminated] / row_levels[illuminated, None]
        mean_level = np.median(row_levels[illuminated])
        yvals = np.nanmedian(relative_rows, axis=0) * mean_level  # Column profile, in the original units
        robust_std = 1.4826 * np.nanmedian(np.abs(relative_rows - np.nanmedian(relative_rows, axis=0)), axis=0) * mean_level
        yerr = 1.2533 * robust_std / np.sqrt(n_illuminated)  # Standard error of the median of each column
        yerr = np.where(yerr > 0, yerr, np.nanmedian(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0)

        # ---------------------- Model ----------------------
        signal_columns = np.isfinite(yvals) & (yvals > low_signal_threshold*np.nanmedian(yvals))  # Columns with enough signal

        # Bad columns are NARROW features of the profile (a few pixels). Each column is compared with the local trend of the profile, given by
        # the mean of the medians of its left and right neighbours (the column itself is excluded, so a bad column cannot hide itself), and
        # not with the smooth model: otherwise, broad (and real) structures of the lamp spectrum that a polynomial cannot follow would be
        # mistaken for bad columns. Bad columns are excluded from the models, so they remain in (and are corrected by) the normalized flat.
        half_window = max(1, bad_column_window // 2)
        padded_profile = np.pad(np.where(np.isfinite(yvals), yvals, np.nanmedian(yvals)), half_window, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded_profile, 2*half_window + 1)
        local_trend = 0.5 * (np.median(windows[:, :half_window], axis=1) + np.median(windows[:, half_window + 1:], axis=1))
        narrow_residuals = (yvals - local_trend) / yerr
        narrow_scatter = 1.4826 * np.median(np.abs(narrow_residuals[signal_columns] - np.median(narrow_residuals[signal_columns])))
        bad_columns = signal_columns & (np.abs(narrow_residuals - np.median(narrow_residuals[signal_columns])) > clip_sigma*narrow_scatter)
        used = signal_columns & ~bad_columns  # Points used in the 1D fit
        model_1d = None
        chi2_reduced = None

        if profile_method == "polynomial":
            # Polynomial.fit performs an ordinary (weighted) least-squares fit in a rescaled domain [-1, 1], which keeps the problem well
            # conditioned even for high degrees. The pixel index (x) has no uncertainty, so OLS (and not ODR) is the appropriate estimator.
            # If the polynomial cannot follow the profile, this shows up in the reduced chi-square (and in the residual plot), instead of
            # being hidden by the rejection of the poorly fitted columns.
            poly = np.polynomial.Polynomial.fit(xvals[used], yvals[used], degree, w=1.0/yerr[used])
            model_1d = poly(xvals)
            n_parameters = degree + 1
            chi2_reduced = float(np.sum(((yvals - model_1d)/yerr)[used]**2) / max(np.count_nonzero(used) - n_parameters, 1))
            model = np.broadcast_to(model_1d, masterflat.shape)

        if profile_method == "median_filter":
            # mode="nearest" repeats the edge values instead of padding with zeros, which would bias the profile close to the edges:
            profile_without_bad_columns = np.where(bad_columns, local_trend, yvals)
            model_1d = ndimage.median_filter(profile_without_bad_columns, size=smooth_window, mode="nearest")
            model = np.broadcast_to(model_1d, masterflat.shape)

        if method == "surface":
            # Step 2: illumination surface. The 1D model removes the lamp spectrum, so the ratio masterflat/model_1d is ~1 and contains only
            # the slit illumination and the vignetting. Fitting this ratio (and not the raw counts) keeps the 2D problem smooth and low-order,
            # and gives the same relative importance to bright and faint parts of the spectrum.
            # The ratio is binned (median of bin_size x bin_size boxes) to make the fit fast and robust, and the surface is then evaluated on
            # the full pixel grid. Coordinates are rescaled to [-1, 1] and Legendre polynomials are used, so the fit is well conditioned.
            nby, nbx = ny // bin_size, nx // bin_size
            if nby < surface_degree_y + 2 or nbx < surface_degree_x + 2:
                raise ValueError("The image is too small for the requested degrees and bin_size. Please reduce bin_size or the degrees.")
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = masterflat / model_1d[None, :]
            ratio[:, ~used] = np.nan  # Columns rejected by the 1D fit (bad columns, very low signal) do not constrain the surface
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # bins made only of excluded columns give NaN and are not used
                binned_ratio = np.nanmedian(ratio[:nby*bin_size, :nbx*bin_size].reshape(nby, bin_size, nbx, bin_size), axis=(1, 3))
                binned_counts = np.nanmedian(masterflat[:nby*bin_size, :nbx*bin_size].reshape(nby, bin_size, nbx, bin_size), axis=(1, 3))
            bin_x = (np.arange(nbx) + 0.5) * bin_size - 0.5
            bin_y = (np.arange(nby) + 0.5) * bin_size - 0.5
            to_unit_x = lambda x: 2.0*x/(nx - 1) - 1.0
            to_unit_y = lambda y: 2.0*y/(ny - 1) - 1.0
            grid_y, grid_x = np.meshgrid(to_unit_y(bin_y), to_unit_x(bin_x), indexing="ij")
            design = np.polynomial.legendre.legvander2d(grid_y.ravel(), grid_x.ravel(), [surface_degree_y, surface_degree_x])
            values = binned_ratio.ravel()
            # Poisson weights: the relative error of a bin scales as 1/sqrt(counts)
            weights = np.sqrt(np.clip(np.nan_to_num(binned_counts.ravel()), 0, None))
            bin_illuminated = np.repeat(illuminated[:nby*bin_size].reshape(nby, bin_size).all(axis=1), nbx)
            fit_used = np.isfinite(values) & bin_illuminated & (weights > 0)
            if np.count_nonzero(fit_used) <= design.shape[1]:
                raise ValueError("Not enough illuminated bins to fit the illumination surface. Please reduce bin_size or the degrees.")
            for _ in range(max_iterations):
                coefficients, *_ = np.linalg.lstsq(design[fit_used] * weights[fit_used, None], values[fit_used] * weights[fit_used], rcond=None)
                residuals = (values - design @ coefficients) * weights
                scatter = 1.4826 * np.median(np.abs(residuals[fit_used] - np.median(residuals[fit_used])))
                new_used = fit_used & (np.abs(residuals - np.median(residuals[fit_used])) < clip_sigma*scatter)
                if np.array_equal(new_used, fit_used):
                    break
                fit_used = new_used
            coefficients = coefficients.reshape(surface_degree_y + 1, surface_degree_x + 1)
            illumination_surface = np.polynomial.legendre.leggrid2d(to_unit_y(np.arange(ny)), to_unit_x(xvals), coefficients)
            model = model_1d[None, :] * illumination_surface

        # ---------------------- Normalization ----------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_master_flat = masterflat / model
        # Pixels that are not corrected (set to 1):
        # - non-finite values or non-positive model;
        # - normalized values outside [low_signal_threshold, 1/low_signal_threshold]: a pixel 10x less (or more) sensitive than the model
        #   is either dead, outside the slit, or a place where the model is unreliable (e.g. an extrapolated polynomial close to zero);
        # - rows outside the slit and, for the fitted methods, columns with too little signal to constrain the 1D model (the polynomial is
        #   only extrapolated there). Columns rejected by sigma clipping (e.g. bad columns) ARE corrected, which is the purpose of the flat.
        low_signal = ~np.isfinite(normalized_master_flat) | (model <= 0) | (normalized_master_flat < low_signal_threshold) \
                     | (normalized_master_flat > 1.0 / low_signal_threshold)
        low_signal |= ~illuminated[:, None]
        if profile_method == "polynomial":
            low_signal |= ~signal_columns[None, :]

        # Relative noise per pixel of the master flat, column by column (median over the illuminated rows, smoothed along x so that the
        # corrected/uncorrected boundary does not flicker), and columns where it exceeds max_flat_noise:
        column_noise = None
        noisy_columns = np.zeros(nx, dtype=bool)
        if masterflat_variance is not None:
            with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                relative_noise = np.sqrt(np.clip(np.asarray(masterflat_variance, dtype=float), 0, None)) / masterflat
                column_noise = np.nanmedian(np.where(masterflat[illuminated] > 0, relative_noise[illuminated], np.nan), axis=0)
            column_noise = ndimage.median_filter(np.nan_to_num(column_noise, nan=np.inf), size=bad_column_window, mode="nearest")
        if max_flat_noise is not None:
            if column_noise is None:
                warnings.warn("'max_flat_noise' was given, but the variance of the master flat is not available, so it was ignored.")
            else:
                noisy_columns = column_noise > max_flat_noise
                # Close small gaps and remove small islands (narrower than bad_column_window), so that the uncorrected region is contiguous:
                structure = np.ones(max(1, bad_column_window), dtype=bool)
                padded = np.pad(noisy_columns, bad_column_window, mode="edge")
                padded = ndimage.binary_opening(ndimage.binary_closing(padded, structure), structure)
                noisy_columns = padded[bad_column_window:-bad_column_window]
                low_signal |= noisy_columns[None, :]
        normalized_master_flat[low_signal] = 1.0
        self.flat_low_signal_mask = low_signal

        # Variance of the normalized flat:
        if masterflat_variance is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_flat_variance = np.asarray(masterflat_variance, dtype=float) / model**2
            norm_flat_variance[low_signal] = 0.0  # These pixels are not corrected, so they add no noise
        else:
            norm_flat_variance = None
            warnings.warn("The variance of the master flat is not available, so the variance of the normalized flat will not be computed.")
        self.norm_flat = normalized_master_flat
        self.norm_flat_variance = norm_flat_variance

        n_rejected = int(np.count_nonzero(bad_columns))  # bad columns (narrow outliers of the profile)
        n_low_signal_columns = int(np.count_nonzero(~signal_columns))
        self.flat_fit_info = {"method": method, "profile_method": profile_method, "degree": degree if profile_method == "polynomial" else None,
                              "surface_degree_y": surface_degree_y if method == "surface" else None,
                              "surface_degree_x": surface_degree_x if method == "surface" else None,
                              "surface_bins_rejected": int(np.count_nonzero(~fit_used)) if method == "surface" else None,
                              "smooth_window": smooth_window if profile_method == "median_filter" else None,
                              "chi2_reduced": chi2_reduced, "n_rejected": n_rejected, "illuminated_rows": illuminated,
                              "column_noise": column_noise, "noisy_columns": noisy_columns}

        good = ~low_signal
        print(f"Normalized master flat: {n_illuminated}/{ny} illuminated rows; median = {np.median(normalized_master_flat[good]):.4f}; "
              f"robust scatter = {1.4826*np.median(np.abs(normalized_master_flat[good] - np.median(normalized_master_flat[good]))):.4f}; "
              f"{np.count_nonzero(low_signal)} pixels set to 1 (low signal).")
        print(f"{n_rejected} bad columns identified (excluded from the model, corrected by the flat)"
              + (f"; {n_low_signal_columns} columns with low signal (left uncorrected)." if profile_method == "polynomial" else "."))
        if column_noise is not None:
            finite_noise = column_noise[np.isfinite(column_noise)]
            print(f"Relative noise per pixel of the master flat: median {100*np.median(finite_noise):.2f}%, from {100*finite_noise.min():.2f}% "
                  f"to {100*finite_noise.max():.2f}% depending on the column.")
        if np.any(noisy_columns):
            edges = np.flatnonzero(np.diff(np.concatenate([[0], noisy_columns.astype(int), [0]])))
            ranges = ", ".join(f"{start}-{end - 1}" for start, end in zip(edges[::2], edges[1::2]))
            print(f"{np.count_nonzero(noisy_columns)} columns left uncorrected because the noise of the flat exceeds max_flat_noise = "
                  f"{100*max_flat_noise:.2f}% (x = {ranges}).")
        if chi2_reduced is not None:
            print(f"Polynomial fit (degree {degree}): reduced chi-square = {chi2_reduced:.2f}.")
            if chi2_reduced > 5:
                warnings.warn("The reduced chi-square of the flat profile fit is large: the polynomial does not describe the profile within its "
                              "uncertainties. Check the residual plot, and consider a higher 'degree' or method='median_filter'.")

        # ---------------------- Saving norm master flat ----------------------
        hdr = fits.getheader(self.flat_list[0],ext=header_hdu_entry) if len(self.flat_list) > 0 else fits.Header()
        for key in ("BZERO", "BSCALE", "BLANK"):
            hdr.remove(key, ignore_missing=True, remove_all=True)
        hdu = fits.PrimaryHDU(normalized_master_flat.astype(np.float32),header=hdr)
        hdu.header['NORMMETH'] = (method, 'Flat normalization method')
        if method == "surface":
            hdu.header['NORMPROF'] = (profile_method, 'Method of the 1D model (step 1 of surface)')
        if profile_method == "polynomial":
            hdu.header['NORMDEGX'] = (degree, 'Polynomial degree along dispersion axis')
        if method == "surface":
            hdu.header['NORMDEGY'] = (surface_degree_y, 'Illumination surface degree along spatial axis')
            hdu.header['NORMSDGX'] = (surface_degree_x, 'Illumination surface degree along dispersion')
        if profile_method == "median_filter":
            hdu.header['NORMWIN'] = (smooth_window, 'Median filter window (pixels)')
        if chi2_reduced is not None:
            hdu.header['NORMCHI2'] = (chi2_reduced, 'Reduced chi-square of the profile fit')
        if max_flat_noise is not None and column_noise is not None:
            hdu.header['NORMMXNS'] = (max_flat_noise, 'Max relative noise of the flat (else set to 1)')
        hdu.header['HISTORY'] = 'Normalized master flat generated with easyspec'
        flat_mask = np.where(low_signal, 2, 0).astype(np.uint8)  # bit 2: flat field not applied (see data_io.MASK_BITS)
        data_io.write_reduced_frame(output_file, normalized_master_flat.astype(np.float32), header=hdu.header, variance=norm_flat_variance,
                                    mask=flat_mask)

        # ---------------------- Diagnostic plots ----------------------
        if plots:
            labels = {"polynomial": f"Polynomial fit (degree {degree})", "median_filter": f"Median filter ({smooth_window} px)"}
            label = labels[profile_method] + (", step 1 of the surface model" if method == "surface" else "")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].set_title("Master flat profile (illuminated rows)")
            axes[0].plot(xvals, yvals, '.', ms=2, alpha=0.5, label="column-median data")
            if np.any(~used):
                axes[0].plot(xvals[~used], yvals[~used], 'x', color='red', label="bad / low-signal columns")
            axes[0].plot(xvals, model_1d, color='limegreen', label=label)
            axes[0].set_xlabel("x (pixels)")
            axes[0].legend()
            axes[1].set_title("Relative residuals (data/model - 1)")
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_residuals = 100*(yvals/model_1d - 1)
                band = 100*yerr/model_1d
            accepted = used & np.isfinite(relative_residuals)
            axes[1].plot(xvals[accepted], relative_residuals[accepted], '.', ms=2, color='limegreen')
            if np.any(~used):
                axes[1].plot(xvals[~used], relative_residuals[~used], 'x', color='red', label="bad / low-signal columns")
            axes[1].fill_between(xvals, -band, band, color='gray', alpha=0.3, label=r"$\pm1\sigma$ of the profile")
            # The y-range follows the accepted points, so that strongly deviant columns do not hide the quality of the fit:
            if np.any(accepted & np.isfinite(relative_residuals)):
                limit = 1.5*np.nanpercentile(np.abs(relative_residuals[accepted]), 99.5)
                limit = max(limit, 1.5*np.nanmax(band[accepted]))
                axes[1].set_ylim(-limit, limit)
            axes[1].axhline(0, color='k', lw=0.8)
            if column_noise is not None:
                noise_axis = axes[1].twinx()
                noise_axis.plot(xvals, 100*column_noise, color='C1', lw=1, label="flat noise per pixel")
                if max_flat_noise is not None:
                    noise_axis.axhline(100*max_flat_noise, color='C1', ls='--', lw=1, label="max_flat_noise")
                    if np.any(noisy_columns):
                        noise_axis.fill_between(xvals, 0, 1, where=noisy_columns, transform=noise_axis.get_xaxis_transform(),
                                                color='C1', alpha=0.12, label="left uncorrected")
                noise_axis.set_ylabel("flat noise per pixel (%)", color='C1')
                noise_axis.set_ylim(0, None)
                noise_axis.legend(loc="lower right", fontsize=8)
            axes[1].set_xlabel("x (pixels)")
            axes[1].set_ylabel("%")
            axes[1].legend()
            for ax in axes:
                ax.grid(which="both",linestyle=":")
            plt.tight_layout()

            if method == "surface":
                # Step 2 diagnostic: slit illumination (masterflat / 1D model) along y, in three ranges of x, compared with the surface.
                plt.figure(figsize=(12, 4))
                plt.title("Illumination along the slit: masterflat / 1D model (points) and illumination surface (lines)")
                rows = np.arange(ny)
                for i, (x1, x2) in enumerate([(0, nx//3), (nx//3, 2*nx//3), (2*nx//3, nx)]):
                    columns = np.zeros(nx, dtype=bool)
                    columns[x1:x2] = True
                    columns &= used
                    if not np.any(columns):
                        continue
                    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        data_profile = np.nanmedian(masterflat[:, columns] / model_1d[None, columns], axis=1)
                    surface_profile = np.median(illumination_surface[:, columns], axis=1)
                    plt.plot(rows[illuminated], data_profile[illuminated], 'o', ms=3, color=f"C{i}", alpha=0.5, mec='none')
                    plt.plot(rows[illuminated], surface_profile[illuminated], '-', color=f"C{i}", label=f"x = {x1}-{x2}")
                plt.xlabel("y (pixels)")
                plt.ylabel("Relative illumination")
                plt.grid(which="both",linestyle=":")
                plt.legend()
                plt.tight_layout()

            image_shape = np.shape(normalized_master_flat)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio))
            ax = plt.gca()
            plt.title('Normalized Master Flat')
            vmin, vmax = np.percentile(normalized_master_flat[good], [0.5, 99.5]) if np.any(good) else (None, None)
            im = ax.imshow(normalized_master_flat, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.08)
            plt.colorbar(im, cax=cax,label="Normalized counts")

            plt.show()

        return normalized_master_flat
    

    def rotate(self,raw_image_data,N_rotations=1):

        """
        Function to rotate the raw data files by N_rotations*90°. In easyspec, the dispersion axis must be in the horizontal.
        
        Parameters
        ----------
        raw_image_data: dict
            A dictionary containing the paths to all data files (dictionary keys) and the raw data for all images (dictionary values). E.g.: the output from the data_paths() function.
        N_rotations: int
            Number of 90° rotations to be performed.

        Returns
        -------
        raw_images_rotated: dict
            A dictionary containing the paths to all data files (dictionary keys) and the rotated raw data for all images (dictionary values).
        """

        raw_images_rotated = {}

        for name,data in raw_image_data.items():
            raw_images_rotated[name] = np.rot90(data,k=N_rotations)
        
        return raw_images_rotated


    def pad(self, image_data, value, x1, x2, y1, y2, plot=True, save_fits=False, get_header_from=None, fits_name="padded_image.fits",
            header_hdu_entry=0, variance=None, mask=None, mask_bit=None):
 
        """
        Replace a rectangular section of an image by a constant value. It is particularly useful when handling master files, e.g. to remove
        from the normalized master flat a feature that exists only in the flats (such as a reflection of the flat-field lamp), by setting
        that region to 1 (i.e. no flat-field correction).
 
        Parameters
        ----------
        image_data: array
            2D array with the image.
        value: float
            The value that replaces the pixels of the section defined by x1, x2, y1, y2.
        x1,x2,y1,y2: integers
            Limits of the section, in pixels (Python convention: image_data[y1:y2, x1:x2]; y = 0 is the bottom row of the plots).
        plot: bool
            If True, easyspec plots the padded image.
        save_fits: bool
            If True, the padded image is saved in the file 'fits_name' (together with its variance and mask, if available).
        get_header_from: string
            Optional. Path to a fits file whose header is copied to the saved file, e.g. "./norm_master_flat.fits".
        fits_name: string
            Name of the fits file to be saved.
        header_hdu_entry: int
            HDU extension of 'get_header_from' where the header is.
        variance: array
            Optional. Variance of image_data. The padded pixels are given zero variance, since a constant has no noise.
        mask: array
            Optional. Data quality bit mask of image_data (see data_io.MASK_BITS). The bit 'mask_bit' is set in the padded pixels.
        mask_bit: int
            Bit of the mask set in the padded pixels. If None (default), bit 2 ("flat field not applied") is used when the padded image is
            the normalized master flat of this session and value = 1; otherwise no bit is set.
 
        Returns
        -------
        image_data: array
            The padded image.
 
        Notes
        -----
        If image_data is the normalized master flat returned by norm_master_flat() in this session, its variance and its map of uncorrected
        pixels are updated automatically, and the padded flat replaces it in this session. In this way, flatten() still recognizes it,
        uses its variance (zero in the padded region) and flags the padded pixels with bit 2 in the masks created by compute_variance().
        """
 
        original = image_data
        image_data = np.array(image_data, dtype=float, copy=True)
        if image_data.ndim != 2:
            raise ValueError(f"'image_data' must be a 2D image, but it has shape {image_data.shape}.")
        ny, nx = image_data.shape
        x1, x2, y1, y2 = (int(round(v)) for v in (x1, x2, y1, y2))
        if not (0 <= x1 < x2 <= nx and 0 <= y1 < y2 <= ny):
            raise ValueError(f"Invalid section x1={x1}, x2={x2}, y1={y1}, y2={y2} for an image with {nx} x {ny} pixels. "
                             f"It must satisfy 0 <= x1 < x2 <= {nx} and 0 <= y1 < y2 <= {ny}.")
        region = np.zeros((ny, nx), dtype=bool)
        region[y1:y2, x1:x2] = True
        image_data[region] = value
 
        # Is this the normalized master flat of this session (see norm_master_flat() and flatten())?
        is_session_flat = original is getattr(self, "norm_flat", None)
        if mask_bit is None and is_session_flat and value == 1:
            mask_bit = 2  # flat field not applied
 
        if variance is None and is_session_flat:
            variance = getattr(self, "norm_flat_variance", None)
        if variance is not None:
            variance = np.array(variance, dtype=float, copy=True)
            if variance.shape != image_data.shape:
                raise ValueError(f"The variance has shape {variance.shape}, but the image has shape {image_data.shape}.")
            variance[region] = 0.0
        if mask is None and is_session_flat and hasattr(self, "flat_low_signal_mask"):
            mask = np.where(self.flat_low_signal_mask, 2, 0).astype(np.uint8)
        if mask is not None:
            mask = np.array(mask, dtype=np.uint8, copy=True)
            if mask.shape != image_data.shape:
                raise ValueError(f"The mask has shape {mask.shape}, but the image has shape {image_data.shape}.")
            if mask_bit is not None:
                mask[region] |= np.uint8(mask_bit)
 
        if is_session_flat:
            # The padded flat replaces the original one in this session, so that flatten() and compute_variance() use it consistently:
            self.norm_flat = image_data
            self.norm_flat_variance = variance
            if value == 1:
                self.flat_low_signal_mask = np.asarray(self.flat_low_signal_mask, dtype=bool) | region if hasattr(self, "flat_low_signal_mask") else region
            print(f"The normalized master flat of this session was updated: pixels x = {x1}-{x2}, y = {y1}-{y2} set to {value}"
                  + (", with zero variance, and flagged as 'flat field not applied' (bit 2)." if value == 1 else ", with zero variance."))
            if value != 1:
                warnings.warn("The normalized master flat was padded with a value different from 1, so the padded pixels WILL be scaled by "
                              f"1/{value} in flatten(). Use value=1 to leave them uncorrected.")
 
        if plot:
            image_shape = np.shape(image_data)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio))
            plt.title('Padded image')
            ax = plt.gca()
            im = ax.imshow(self._log_display(image_data), origin='lower', cmap='gray')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.08)
            plt.colorbar(im, cax=cax,label="Counts in log scale")
            plt.show()
 
        if save_fits:
            hdr = fits.getheader(get_header_from, ext=header_hdu_entry) if get_header_from is not None else None
            if hdr is not None:
                hdr = hdr.copy()
            else:
                hdr = fits.Header()
            hdr['HISTORY'] = f'Padded with easyspec: x = {x1}-{x2}, y = {y1}-{y2} set to {value}'
            data_io.write_reduced_frame(fits_name, image_data, header=hdr, variance=variance, mask=mask)
            print(f"Padded image saved in {fits_name}.")
 
        return image_data

    def plot(self,image_data,figure_name,save=True,format="png"):

        """
        This function plots and saves a single image.
        
        Parameters
        ----------
        image_data: array
            Array containing the raw data for one image.
        figure_name: string
            Name of the figure. It can also be a path ending with the figure name, e.g. ./output/FIGURE_NAME, as long as this directory exists.
        save: bool
            If True, the image will be saved.
        format: string
            You can use png, jpeg, pdf and all other formats accepted by matplotlib.

        Returns
        -------
        figure_name.png : file
            An image saved in the current directory.
        """
        
        image_shape = np.shape(image_data)
        aspect_ratio = image_shape[0]/image_shape[1]
        plt.figure(figsize=(12,12*aspect_ratio)) 
        plt.title(figure_name)
        ax = plt.gca()
        im = ax.imshow(self._log_display(image_data), origin='lower', cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.08)
        plt.colorbar(im, cax=cax,label="Counts in log scale")

        if save:
            plt.savefig(figure_name+"."+format,bbox_inches='tight')
        
        plt.show()


    def flatten(self,debiased_data,norm_master_flat,Type="all",norm_master_flat_variance=None):
 
        """
        Function to apply the master flat on the rest of the data.
 
        Parameters
        ----------
        debiased_data: dict
            Dictionary containing debiased data. It can even contain the raw bias and dark files, but they will be ignored in the process.
        norm_master_flat: array
            Array with normalized master flat raw data.
        Type: string
            Type of data files to apply the normalized master flat. Options are: "all", "lamp", "standard_star", and "target".
        norm_master_flat_variance: array
            Variance of the normalized master flat. If None and norm_master_flat is the array returned by norm_master_flat() in this session,
            easyspec uses the variance computed by that function. If you loaded the normalized flat from a file, you can get its variance
            with read_master().
 
        Returns
        -------
        flattened_debiased_data_out: dict
            Dictionary containing the flattened data.
 
        Notes
        -----
        The flat applied, its variance and the list of flattened files are stored in the attribute 'self.flat_applied', which is used by
        compute_variance().
        """
 
        valid_types = ["all", "lamp", "standard_star", "target"]
        if Type not in valid_types:
            raise ValueError(f"Invalid Type '{Type}'. Options are: " + ", ".join(f'"{t}"' for t in valid_types) + ".")
 
        # If the flat comes from norm_master_flat() in this session, we can recover its variance and its uncorrected (low-signal) pixels:
        from_this_session = norm_master_flat is getattr(self, "norm_flat", None)
        if norm_master_flat_variance is None and from_this_session:
            norm_master_flat_variance = getattr(self, "norm_flat_variance", None)
        uncorrected = np.array(self.flat_low_signal_mask, copy=True) if from_this_session and hasattr(self, "flat_low_signal_mask") \
                      else np.zeros(np.shape(norm_master_flat), dtype=bool)
 
        flatten_list = []
        for file_name in debiased_data.keys():
            if Type == "all" and file_name not in self.bias_list and file_name not in self.darks_list and file_name not in self.flat_list:
                flatten_list = flatten_list + [file_name]
            elif Type == "lamp" and file_name in self.lamp_list:
                flatten_list = flatten_list + [file_name]
            elif Type == "standard_star" and file_name in self.std_list:
                flatten_list = flatten_list + [file_name]
            elif Type == "target" and file_name in self.target_list:
                flatten_list = flatten_list + [file_name]
 
        flattened_debiased_data_out = {}  # Dictionary for the flattened images
 
        # This block is useful only for norm_master_flats which were not created by easyspec:
        norm_master_flat = np.asarray(norm_master_flat, dtype=float)
        invalid = ~np.isfinite(norm_master_flat) | (norm_master_flat <= 0)
        if np.any(invalid):
            warnings.warn(f"The normalized master flat has {np.count_nonzero(invalid)} pixels with zero, negative or non-finite values. "
                          "They will be set to 1 (no flat-field correction) to avoid divisions by zero.")
            norm_master_flat = np.where(invalid, 1.0, norm_master_flat)
        uncorrected |= invalid
 
        if norm_master_flat_variance is not None:
            norm_master_flat_variance = np.where(uncorrected, 0.0, np.asarray(norm_master_flat_variance, dtype=float))
 
        for i in range(len(flatten_list)):
            flattened_debiased_data_out[flatten_list[i]] = debiased_data[flatten_list[i]]/norm_master_flat  # Flattening the images with the normalized master flat.
 
        self.flat_applied = {"flat": norm_master_flat, "variance": norm_master_flat_variance, "uncorrected": uncorrected,
                             "files": set(flatten_list)}
 
        return flattened_debiased_data_out
    
    # Bits of the data quality mask created by compute_variance() and saved by save_fits_files():
    MASK_BITS = data_io.MASK_BITS  # Bits of the data quality mask, defined in easyspec.data_io
 
 
    def compute_variance(self, electron_data, Type="all", readnoise=None, readnoise_header_entry="RDNOISE", masterbias_variance=None,
                         masterdark=None, masterdark_variance=None, verbose=True):
 
        """
        Function to compute the variance (in electrons squared) and the data quality mask of each reduced image.
 
        All reduction steps up to gain_correct() are pixel-by-pixel linear operations, so the variance of the final image can be written in
        closed form (the "CCD equation"). For a pixel with final value S (in electrons), gain g, read noise RN (in electrons), master bias B,
        bias-subtracted master dark D (scaled by s = t_image/t_dark, see sub_dark()), and normalized flat F:
 
            S = g * (R - B - s*D) / F,
 
            Var(S) = [ max(S*F + g*s*D, 0) + RN^2 + g^2 * ((1 - s)^2 * Var(B) + s^2 * Var(D)) ] / F^2  +  S^2 * Var(F) / F^2.
 
        The first term is the Poisson noise of the photo-electrons and of the dark current, the second is the read noise, the third is the
        noise of the master bias and master dark (the factor (1 - s)^2 accounts for the fact that the same master bias was also subtracted
        from the darks), and the last one is the noise of the flat field. All these ingredients are taken automatically from the previous
        steps of this session (master(), sub_dark(), norm_master_flat(), flatten(), gain_correct(), remove_cosmic_rays()); if you loaded
        the masters from files, you can pass them explicitly (see read_master()).
 
        IMPORTANT: this function must be applied to the output of gain_correct() or remove_cosmic_rays(), i.e., BEFORE align(), so that the
        images have the same geometry as the masters.
 
        Parameters
        ----------
        electron_data: dict
            Dictionary with the reduced images in electrons (e.g. the output of gain_correct() or remove_cosmic_rays()).
        Type: string
            Type of data files. Options are: "all", "lamp", "standard_star", and "target".
        readnoise: float
            CCD read noise in electrons. If None, easyspec uses the value used by remove_cosmic_rays() for each file or, if not available,
            looks for it in the header of each file.
        readnoise_header_entry: string
            Header keyword with the read noise, used only if readnoise=None.
        masterbias_variance: array
            Variance of the master bias (ADU^2). If None, the one computed by master("bias", ...) in this session is used.
        masterdark: array
            Bias-subtracted master dark (ADU). If None, the one computed by master("dark", ...) in this session is used. Only needed for the
            frames from which a dark was subtracted (see sub_dark()).
        masterdark_variance: array
            Variance of the master dark (ADU^2). If None, the one computed by master("dark", ...) in this session is used.
        verbose: bool
            If True, easyspec prints a summary for each image.
 
        Returns
        -------
        variance: dict
            Dictionary with the variance of each image, in electrons squared.
        mask: dict
            Dictionary with the data quality mask of each image (unsigned integer bit mask, 0 = good pixel). The meaning of each bit is given
            in the attribute 'cleaning.MASK_BITS': 1 = cosmic ray, 2 = flat field not applied, 4 = non-finite value.
 
        Notes
        -----
        The part of the variance due to the calibration masters (the terms with Var(B), Var(D) and Var(F)) is the same random error in all
        frames reduced with the same masters, so it does not average down when the frames are combined. This part is stored separately in
        the attribute 'self.calibration_variance' (a dictionary with the same keys), which is used by master() to combine the variances
        correctly, and transformed by align() together with the variance frames.
 
        Uncertainties of the smooth model used to normalize the flat and correlations introduced later by sub-pixel resampling (see align())
        are not included.
        """
 
        selected = self._select_for_gain_and_cr(electron_data, Type, "compute_variance")
 
        gain_used = getattr(self, "gain_used", {})
        not_converted = [Path(file_name).name for file_name in selected if file_name not in gain_used]
        if not_converted:
            raise RuntimeError("The following images were not converted to electrons: " + ", ".join(not_converted) +
                               ". Please run gain_correct() on them before compute_variance().")
 
        masters = getattr(self, "masters", {})
        master_variances = getattr(self, "master_variances", {})
        if masterbias_variance is None:
            masterbias_variance = master_variances.get("bias")
        if masterdark is None:
            masterdark = masters.get("dark")
        if masterdark_variance is None:
            masterdark_variance = master_variances.get("dark")
        dark_scale_factors = getattr(self, "dark_scale_factors", {})
        flat_applied = getattr(self, "flat_applied", None)
        cosmic_ray_masks = getattr(self, "cosmic_ray_masks", {})
        readnoise_used = getattr(self, "readnoise_used", {})
 
        missing_terms = set()
        variance = {}
        mask = {}
        calibration_variance_dict = {}
        for file_name in selected:
            S = np.asarray(electron_data[file_name], dtype=float)
            shape = S.shape
            g = gain_used[file_name]
 
            if readnoise is not None:
                rn = float(readnoise)
            elif file_name in readnoise_used:
                rn = readnoise_used[file_name]
            else:
                rn = float(self.look_in_header(file_name, readnoise_header_entry))
 
            # Ingredients (each one is checked against the image geometry):
            def checked(array, name):
                array = np.asarray(array, dtype=float)
                if array.shape != shape and array.shape != ():
                    raise RuntimeError(f"The {name} has shape {array.shape}, but {Path(file_name).name} has shape {shape}. compute_variance() "
                                       "must be applied before align(), to images with the same geometry as the masters.")
                return array
 
            s = float(dark_scale_factors.get(file_name, 0.0))  # s = 0 means that no dark was subtracted from this image
            if flat_applied is not None and file_name in flat_applied["files"]:
                F = checked(flat_applied["flat"], "normalized flat")
                VF = flat_applied["variance"]
                VF = checked(VF, "variance of the normalized flat") if VF is not None else None
                if VF is None:
                    missing_terms.add("variance of the normalized flat")
                uncorrected = checked(flat_applied["uncorrected"], "flat mask").astype(bool)
            else:
                F, VF, uncorrected = 1.0, None, np.zeros(shape, dtype=bool)
 
            # Electrons actually collected (photo-electrons + dark current), which define the Poisson noise:
            collected = S * F
            calibration_variance = np.zeros(shape)  # In ADU^2
            if masterbias_variance is not None:
                calibration_variance = calibration_variance + (1.0 - s)**2 * checked(masterbias_variance, "master bias variance")
            else:
                missing_terms.add("variance of the master bias")
            if s != 0.0:
                if masterdark is not None:
                    collected = collected + g * s * checked(masterdark, "master dark")
                else:
                    missing_terms.add("master dark (Poisson noise of the dark current)")
                if masterdark_variance is not None:
                    calibration_variance = calibration_variance + s**2 * checked(masterdark_variance, "master dark variance")
                else:
                    missing_terms.add("variance of the master dark")
 
            # Noise that is independent from frame to frame (photon noise and read noise of this exposure):
            independent_variance = (np.clip(collected, 0, None) + rn**2) / F**2
            # Noise of the calibration masters (bias, dark, flat). The same masters are applied to all frames, so this part is common to
            # (i.e. fully correlated between) all frames and does not average down when frames are combined (see master()):
            common_variance = g**2 * calibration_variance / F**2
            if VF is not None:
                common_variance = common_variance + S**2 * VF / F**2
            common_variance = np.broadcast_to(common_variance, shape).astype(float)
            file_variance = np.broadcast_to(independent_variance, shape).astype(float) + common_variance
            calibration_variance_dict[file_name] = common_variance
 
            # Data quality mask:
            file_mask = np.zeros(shape, dtype=np.uint8)
            if file_name in cosmic_ray_masks and np.shape(cosmic_ray_masks[file_name]) == shape:
                file_mask[np.asarray(cosmic_ray_masks[file_name], dtype=bool)] |= 1
            file_mask[uncorrected] |= 2
            file_mask[~np.isfinite(S) | ~np.isfinite(file_variance)] |= 4
 
            variance[file_name] = file_variance
            mask[file_name] = file_mask
 
            if verbose:
                good = file_mask == 0
                median_snr = np.median(S[good] / np.sqrt(file_variance[good])) if np.any(good) else np.nan
                print(f"{Path(file_name).name}: gain = {g} e-/ADU, read noise = {rn} e-, dark scale = {s:g}; "
                      f"median S/N per pixel = {median_snr:.2f}; {np.count_nonzero(~good)} flagged pixels.")
 
        if missing_terms:
            warnings.warn("The following ingredients were not available, so their contribution to the variance was neglected: "
                          + "; ".join(sorted(missing_terms)) + ". The uncertainties may be underestimated.")
 
        self.variance = variance
        self.data_quality = mask
        if not hasattr(self, "calibration_variance"):
            self.calibration_variance = {}
        self.calibration_variance.update(calibration_variance_dict)
        return variance, mask
 
 
    @staticmethod
    def read_master(file_name):
 
        """
        Read a master (or normalized master flat) file created by easyspec, returning its data and, if available, its variance.
 
        Parameters
        ----------
        file_name: string
            Path to the fits file, e.g. "master_bias.fits" or "norm_master_flat.fits".
 
        Returns
        -------
        data: array
            The master data (primary HDU).
        variance: array or None
            The variance stored in the "VAR" extension, or None if the file has no such extension (e.g. files created by older easyspec versions).

        Notes
        -----
        This is a shortcut to easyspec.data_io.read_master(), kept here for convenience.
        """
 
        return data_io.read_master(file_name)
 
 
    def save_fits_files(self,datacube,output_directory,header_hdu_entry=0,Type="all",variance=None,mask=None):
 
        """
        This function saves the data contained in the dictionary 'datacube' as fits files in the directory 'output_directory'.
 
        The data are saved in the primary HDU (so fits.getdata() keeps working as before). If 'variance' and/or 'mask' are given (e.g. the
        outputs of compute_variance()), they are saved in two extra extensions called "VAR" (variance) and "MASK" (data quality bit mask,
        0 = good pixel; the meaning of each bit is written in the header of the MASK extension). See the module easyspec.data_io for the
        file format, and easyspec.data_io.read_reduced_frame() to read these files back.
 
        Parameters
        ----------
        datacube: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data for their respective images (dictionary values).
        output_directory: string
            String with the path to the output directory. It is created if it does not exist.
        header_hdu_entry: int
            HDU extension where we can find the header. Please check your data before choosing this value.
        Type: string
            Type of data files to be saved. Options are: "all", "bias", "flat", "lamp", "standard_star", "target", and "dark".
        variance: dict
            Optional. Dictionary with the variance of each image (same keys as datacube).
        mask: dict
            Optional. Dictionary with the data quality mask of each image (same keys as datacube).
 
        Returns
        -------
            Saves one or more fits files in the output directory.
 
        """
 
        if Type == "all":
            imagenames = list(datacube.keys())
        elif Type == "bias":
            imagenames = self.bias_list
        elif Type == "flat":
            imagenames = self.flat_list
        elif Type == "lamp":
            imagenames = self.lamp_list
        elif Type == "standard_star":
            imagenames = self.std_list
        elif Type == "target":
            imagenames = self.target_list
        elif Type == "dark":
            imagenames = self.darks_list
        else:
            raise ValueError('Invalid variable Type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')
 
        output_directory = Path(output_directory).resolve()
        in_electrons = getattr(self, "gain_used", {})

        for n,file_name in enumerate(imagenames):
            hdr = fits.getheader(file_name,ext=header_hdu_entry)
            hdr['COMMENT'] = 'File generated with easyspec'
            data_io.write_reduced_frame(output_directory / f"{Type}_{n:03d}.fits", datacube[file_name], header=hdr,
                                        variance=None if variance is None else variance.get(file_name),
                                        mask=None if mask is None else mask.get(file_name),
                                        units="electron" if file_name in in_electrons else None)

    @staticmethod
    def _log_display(image_data):

        """
        Return log10(image_data) for display purposes only, without warnings or NaNs caused by zero or negative pixels.

        Pixels below a floor value are set to this floor before taking the logarithm. If the image has no zero or negative
        pixels, the result is identical to np.log10(image_data). Otherwise, the floor is the robust noise level of the image
        (normalized median absolute deviation), so that noise-dominated pixels, which scatter around zero after the bias and
        dark subtraction, appear as a uniform dark background instead of blank (NaN) pixels.

        The input data are never modified: this function is meant to be used only inside plotting functions.

        Parameters
        ----------
        image_data: array
            Image to be displayed.

        Returns
        -------
        log_image: array
            log10 of the image, with all values below the floor set to log10(floor).
        """

        image_data = np.asarray(image_data, dtype=float)
        finite = image_data[np.isfinite(image_data)]
        if finite.size == 0 or np.all(finite > 0):
            return np.log10(image_data)

        noise = 1.4826 * np.median(np.abs(finite - np.median(finite)))  # Robust standard deviation
        positive = finite[finite > 0]
        if noise > 0:
            floor = noise
        elif positive.size > 0:
            floor = positive.min()
        else:
            floor = 1.0  # The image has no positive pixels at all
        return np.log10(np.clip(image_data, floor, None))

    def look_in_header(self,file_name,I_am_looking_for):

        """
        This function looks for a specific entry over all header entries in a fits file.

        Parameters
        ----------
        file_name: string
            The path to the fits data file.
        I_am_looking_for: string
            Which entry are you looking for? E.g. "EXPTIME" or "RDNOISE".

        Returns
        -------
        value: don't have a predefined type
            The value corresponding to the key you are looking for.
        
        """

        ref_hdul = fits.open(file_name)
        value = None
        for extension in range(len(ref_hdul)):
            try:
                value = ref_hdul[extension].header[I_am_looking_for]
                if value is not None:
                    break
            except:
                pass
        
        if value is None:
            raise RuntimeError("No entry called "+I_am_looking_for+" was found in the fits file.")
        
        return value

    def _select_for_gain_and_cr(self, data, Type, function_name):

        """Helper used by gain_correct() and remove_cosmic_rays(): returns the keys of 'data' that match 'Type'."""

        valid_types = ["all", "lamp", "standard_star", "target"]
        if Type not in valid_types:
            raise ValueError(f"Invalid Type '{Type}'. Options are: \"all\", \"lamp\", \"standard_star\", and \"target\".")

        selected = []
        for file_name in data.keys():
            if Type == "all" and file_name not in self.bias_list and file_name not in self.darks_list and file_name not in self.flat_list:
                selected.append(file_name)
            elif Type == "target" and file_name in self.target_list:
                selected.append(file_name)
            elif Type == "standard_star" and file_name in self.std_list:
                selected.append(file_name)
            elif Type == "lamp" and file_name in self.lamp_list:
                selected.append(file_name)

        if len(selected) == 0:
            raise ValueError(f"{function_name}: no image of Type '{Type}' was found in the input dictionary. Please be sure that you ran data_paths() "
                             "and that you did not change the keys in the input dictionary.")
        return selected


    def gain_correct(self, data, gain=None, gain_header_entry="GAIN", Type="all", verbose=True):

        """
        Function to convert the image units from ADU to electrons, multiplying each image by the detector gain.

        Parameters
        ----------
        data: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data in ADU (dictionary values), e.g. the output of flatten().
        gain: float
            This is the CCD gain (e-/ADU). If gain = None, easyspec will look for this entry automatically in the header of EACH file, so frames taken
            with different detector settings (e.g. lamp and science) are converted with their own gain.
        gain_header_entry: string
            We use this only if gain=None. This must be the keyword containing the gain value in the fits file header.
        Type: string
            Type of data files to convert. Options are: "all", "lamp", "standard_star", and "target".
        verbose: bool
            If True, easyspec prints the gain used for each file.

        Returns
        -------
        electron_data: dict
            Dictionary containing the data in electrons.

        Notes
        -----
        The gain used for each file is stored in the attribute 'self.gain_used' (a dictionary with the file paths as keys). It is also used by
        remove_cosmic_rays() to confirm that the data are already in electrons. Please apply this function only once to each image.
        """

        selected = self._select_for_gain_and_cr(data, Type, "gain_correct")

        if not hasattr(self, "gain_used"):
            self.gain_used = {}

        electron_data = {}
        for file_name in selected:
            file_gain = gain if gain is not None else float(self.look_in_header(file_name,gain_header_entry))
            if not file_gain > 0:
                raise ValueError(f"Invalid gain ({file_gain}) for {Path(file_name).name}. The gain must be positive.")
            self.gain_used[file_name] = file_gain

            image = CCDData(np.asarray(data[file_name], dtype=float), unit='adu')
            electron_data[file_name] = np.asarray(ccdp.gain_correct(image, file_gain * u.electron / u.adu).data)  # From ADU to electrons

            if verbose:
                print(f"{Path(file_name).name}: converted from ADU to electrons (gain = {file_gain} e-/ADU).")

        return electron_data


    def remove_cosmic_rays(self, electron_data, readnoise=None, readnoise_header_entry="RDNOISE", Type="all", sigclip=5, remove_cr_from_lamps=False,
                           saturation_level=65535, verbose=True, **lacosmic_kwargs):

        """
        Function to remove cosmic rays from target and standard star data based on the LACosmic method originally developed and implemented for IRAF
        by Pieter G. van Dokkum. The data must be in electrons, i.e., you must run gain_correct() before this function.

        Parameters
        ----------
        electron_data: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data in electrons (dictionary values), e.g. the output of
            gain_correct().
        readnoise: float
            This is the CCD read noise (in electrons), used to generate the noise model of the image. If readnoise = None, easyspec will look for this
            entry automatically in the header of EACH file.
        readnoise_header_entry: string
            We use this only if readnoise=None. This must be the keyword containing the read noise value in the fits file header.
        Type: string
            Type of data files to clean. Options are: "all", "lamp", "standard_star", and "target".
        sigclip: float
            Laplacian-to-noise limit for cosmic ray detection. Lower values will flag more pixels as cosmic rays.
        remove_cr_from_lamps: bool
            If False (default), lamp images are returned unchanged, without cosmic ray removal. Strong and narrow arc lines can be mistaken for
            cosmic rays by LACosmic, and lamp exposures are usually short, so cosmic rays are rare. Set it to True to also clean the lamp images.
        saturation_level: float
            Saturation level of the detector, in ADU (it is converted to electrons internally with the gain stored by gain_correct()). Pixels at or
            above this level are protected from being flagged as cosmic rays (e.g. saturated stars). Default = 65535 (16-bit detectors).
        verbose: bool
            If True, easyspec prints the number of pixels flagged as cosmic rays in each image.
        **lacosmic_kwargs:
            Any other parameter accepted by ccdproc.cosmicray_lacosmic, e.g. objlim, sigfrac, niter, psffwhm. Use them to fine-tune the detection.
            For instance, increasing objlim makes LACosmic less likely to flag the peaks of narrow emission lines or compact sources.

        Returns
        -------
        CR_corrected_data: dict
            Dictionary containing the data (in electrons) corrected for cosmic rays. Lamps not cleaned are included unchanged.

        Notes
        -----
        The boolean masks of the pixels flagged as cosmic rays (True = cosmic ray) are stored in the attribute 'self.cosmic_ray_masks', a dictionary
        with the same keys as CR_corrected_data. Lamps not cleaned for cosmic rays get an all-False mask. The read noise used for each file is stored in
        'self.readnoise_used'. Please note that these masks follow the geometry of the images at this step: if you later trim, rotate or align the
        images, the masks must receive the same operations.
        """

        selected = self._select_for_gain_and_cr(electron_data, Type, "remove_cosmic_rays")

        # LACosmic needs the data in electrons. gain_correct() records every file it converts, so we can check it:
        gain_used = getattr(self, "gain_used", {})
        not_converted = [Path(file_name).name for file_name in selected if file_name not in gain_used]
        if not_converted:
            raise RuntimeError("The following images were not converted to electrons: " + ", ".join(not_converted) +
                               ". Please run gain_correct() on them before remove_cosmic_rays().")

        CR_corrected_data = {}
        self.cosmic_ray_masks = {}
        self.readnoise_used = {}
        for file_name in selected:
            image_data = np.asarray(electron_data[file_name], dtype=float)

            if file_name in self.lamp_list and not remove_cr_from_lamps:
                CR_corrected_data[file_name] = image_data.copy()
                self.cosmic_ray_masks[file_name] = np.zeros(image_data.shape, dtype=bool)
                if verbose:
                    print(f"{Path(file_name).name}: lamp, no cosmic ray removal.")
                continue

            file_readnoise = readnoise if readnoise is not None else float(self.look_in_header(file_name,readnoise_header_entry))
            self.readnoise_used[file_name] = file_readnoise

            # The data are already in electrons, so LACosmic must not apply the gain again (gain=1, gain_apply=False),
            # and the saturation level must also be given in electrons.
            image = CCDData(image_data, unit='electron')
            image = ccdp.cosmicray_lacosmic(image, readnoise=file_readnoise, sigclip=sigclip, gain=1.0, gain_apply=False,
                                            satlevel=saturation_level*gain_used[file_name], **lacosmic_kwargs)

            CR_corrected_data[file_name] = np.asarray(image.data)  # Cleaned data (flagged pixels replaced by LACosmic), back to a numpy array.
            self.cosmic_ray_masks[file_name] = np.asarray(image.mask, dtype=bool)  # True = pixel flagged as cosmic ray

            if verbose:
                n_flagged = int(self.cosmic_ray_masks[file_name].sum())
                print(f"{Path(file_name).name}: {n_flagged} pixels flagged as cosmic rays ({100*n_flagged/image.size:.3f}% of the image).")

        return CR_corrected_data


    def CR_and_gain_corrections(self,flattened_debiased_data,gain=None,gain_header_entry="GAIN",readnoise=None,readnoise_header_entry="RDNOISE",Type="all",sigclip=5,
                                remove_cr_from_lamps=False,saturation_level=65535,verbose=True,**lacosmic_kwargs):

        """
        Shortcut that runs gain_correct() followed by remove_cosmic_rays(). It is kept for backward compatibility with previous easyspec versions.
        Please see the documentation of these two functions for a description of the parameters.

        Returns
        -------
        CR_corrected_data: dict
            Dictionary containing the data corrected for gain (now in electrons) and cosmic rays.
        """

        electron_data = self.gain_correct(flattened_debiased_data, gain=gain, gain_header_entry=gain_header_entry, Type=Type, verbose=verbose)
        return self.remove_cosmic_rays(electron_data, readnoise=readnoise, readnoise_header_entry=readnoise_header_entry, Type=Type, sigclip=sigclip,
                                       remove_cr_from_lamps=remove_cr_from_lamps, saturation_level=saturation_level, verbose=verbose,
                                       **lacosmic_kwargs)

    def align(self, data, Type="all", method="auto", reference=None, allow_rotation=True, allow_scale=False,
              subpixel=True, interpolation_order=3, crop=True, fill_value=0.0, fwhm=4.0, detection_sigma=5.0,
              max_stars=60, match_tolerance=2.0, min_matches=6, median_prefilter=3, plot=False, verbose=True,
              variance=None, mask=None, variance_mode="aperture"):
 
        """
        Align target and/or standard-star images, allowing for translation and rotation (and, optionally, a change
        of plate scale). It works both for photometric images (star fields) and for long-slit spectra.
 
        Images are grouped by their parent directory (one group per target, plus one group for the standard star),
        and every image in a group is resampled onto the pixel grid of the group's reference image.
 
        Two methods are available to estimate the geometric transformation between each image and its reference:
 
        - "stars": point sources are detected in both images and matched through triangle asterisms, which are
          invariant to translation, rotation and scale. The transformation is then refined with a least-squares
          fit to all matched stars. This is the most accurate option for photometric data.
        - "correlation": Fourier-based registration. The rotation angle is obtained from the polar-resampled
          Fourier amplitude spectra (Fourier-Mellin approach, which is insensitive to translation), and the
          translation from phase correlation with sub-pixel peak interpolation. This option does not need stars,
          so it is the natural choice for spectra (a single trace, extended sources, etc.).
        - "auto" (default): tries "stars" and falls back to "correlation" whenever not enough stars are matched or
          the matched sources are nearly collinear (as happens with emission knots along a spectral trace).
 
        Lamp, bias, dark and flat images are never aligned, but if they are present in 'data' they are cropped
        together with the aligned images (when crop=True), so that all frames keep the same geometry. This is
        important in spectroscopy, where the lamp must stay on the same pixel grid as the science frames.
 
        Parameters
        ----------
        data: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data for their
            respective images (dictionary values), e.g. the output of CR_and_gain_corrections().
        Type: string
            Type of data files to align. Options are: "all" (targets and standard star), "target", and
            "standard_star".
        method: string
            Method used to estimate the transformation. Options are: "auto", "stars", and "correlation".
        reference: None, int, or string
            Reference image of each group. If None, the first image (in alphabetical order) of each group is used.
            If an integer, it is the index of the reference image inside each group. If a string, it must be the
            path to one of the images; this image becomes the reference of its own group, while the other groups
            use their first image.
        allow_rotation: bool
            If False, only translations are fitted.
        allow_scale: bool
            If True, a change of plate scale (isotropic magnification) is also fitted. Use it only when combining
            images from different instruments/binnings; otherwise keep it False.
        subpixel: bool
            If True (default), images are resampled with a spline of order 'interpolation_order'. If False, only
            integer-pixel translations are applied (no interpolation at all, so pixel values and noise properties
            are preserved). In this case rotation and scale are ignored.
        interpolation_order: int
            Order of the spline interpolation (0 to 5) used to resample the images when subpixel=True. Order 1
            (bilinear) slightly smooths the noise; order 3 (bicubic, default) preserves the PSF shape better but may
            produce small ringing around very sharp features (e.g. residual cosmic rays or hot pixels).
        crop: bool
            If True, all images in 'data' are cropped to the largest rectangle that contains valid pixels in every
            aligned image. If False, the original shape is kept and pixels falling outside the original field of
            view are set to 'fill_value'.
        fill_value: float
            Value assigned to pixels without data when crop=False. Use np.nan if your subsequent steps are NaN-aware.
        fwhm: float
            Approximate full width at half maximum of the point sources, in pixels. Used only by the "stars" method.
        detection_sigma: float
            Detection threshold, in units of the background noise of the smoothed image. Used only by "stars".
        max_stars: int
            Maximum number of (brightest) stars used for the matching. Used only by "stars".
        match_tolerance: float
            Maximum distance, in pixels, between a transformed star and its counterpart in the reference image for
            them to be considered a match. Used only by "stars".
        min_matches: int
            Minimum number of matched stars required to accept a "stars" solution.
        median_prefilter: int or None
            Size (in pixels) of a median filter applied to copies of the images before estimating the
            transformation, to suppress hot pixels, residual cosmic rays and other single-pixel defects that would
            otherwise dominate the star detection and the Fourier correlation (they do not move with the field).
            The filtered copies are used only for the estimate; the output images are never filtered.
            Must be smaller than the FWHM of the sources. Use None or 0 to disable it.
        plot: bool
            If True, easyspec plots, for each aligned image, a color composite of the reference image (magenta) and
            the aligned image (green). Well-aligned sources appear white/gray; misalignments show colored doublets.
        verbose: bool
            If True, easyspec prints the transformation found for each image.
        variance: dict
            Optional. Variance frames of the images in 'data' (e.g. the output of compute_variance()), with the geometry they had BEFORE
            the alignment. They receive the same transformation and crop as the images. Entries of images that are not aligned (e.g. lamps)
            are only cropped.
        mask: dict
            Optional. Data quality bit masks of the images in 'data' (e.g. the output of compute_variance()). They receive the same
            transformation and crop as the images: an aligned pixel inherits the bits of the input pixels that dominate its interpolation.
            If crop=False, pixels outside the original field of view get bit 8 (no data).
        variance_mode: string
            How the variance is resampled when subpixel=True (see the Notes). Options are "aperture" (default) and "pixel".
 
        Returns
        -------
        aligned_data: dict
            Dictionary with the same keys as 'data', containing the aligned (and possibly cropped) images.
        aligned_variance, aligned_mask: dict
            Only returned if 'variance' and/or 'mask' are given (as None if the corresponding input is None). In this case the function
            returns the tuple (aligned_data, aligned_variance, aligned_mask).
 
        Notes
        -----
        The transformation found for each image is stored in the attribute 'self.alignment_transforms', a dictionary
        whose keys are the file paths. Each entry contains the 2x2 matrix A and the offset t that map the pixel
        coordinates (x, y) = (column, row) of the image onto the reference image, p_ref = A @ p_img + t, plus the
        rotation angle (in degrees, counterclockwise when displayed with origin='lower'), the scale, the shift of the
        image center, the method used and, for the "stars" method, the number of matched stars and the rms of the
        residuals. The crop box (x1, x2, y1, y2) is stored in 'self.alignment_crop_box'.
 
        Sub-pixel resampling (subpixel=True) turns each aligned pixel into a weighted sum of input pixels, so the noise of neighbouring
        pixels becomes correlated. No single variance frame can then be exact both for individual pixels and for sums of pixels:
 
        - variance_mode="aperture" (default): the variance is resampled like a flux (bilinear weights), so the total variance is conserved.
          It is accurate for sums over apertures several pixels wide in both directions (e.g. within ~2% for an 11x11 photometric aperture
          with the default bicubic interpolation) and conservative (overestimated) for single pixels and very narrow apertures (e.g. ~+19%
          in variance for the sum along one spectral column).
        - variance_mode="pixel": exact variance of each individual pixel (sum of the squared interpolation weights), but sums of these
          values underestimate the variance of sums of pixels (by ~15-30% with bicubic interpolation).
 
        With subpixel=False (integer shifts, no resampling) the noise stays uncorrelated and the propagated variance is exact in all cases.
        This is therefore the recommended option for spectra, which usually need only small vertical shifts.
        """
 
        valid_types = ["all", "target", "standard_star"]
        if Type not in valid_types:
            raise ValueError(f"Invalid Type '{Type}'. Options are: " + ", ".join(f'"{t}"' for t in valid_types) + ".")
        if variance_mode not in ("aperture", "pixel"):
            raise ValueError(f"Invalid variance_mode '{variance_mode}'. Options are: 'aperture' and 'pixel'.")
 
        candidates = []
        if Type in ("all", "target"):
            candidates += list(self.target_list)
        if Type in ("all", "standard_star"):
            candidates += list(self.std_list)
        align_list = [name for name in candidates if name in data]
        if len(align_list) == 0:
            raise ValueError(f"No image of Type '{Type}' was found in 'data'. Please check that you ran data_paths() "
                             "and that the dictionary keys were not changed.")
 
        # Group the images by parent directory (one group per target or standard star):
        groups = {}
        for name in align_list:
            groups.setdefault(str(Path(name).parent.resolve()), []).append(name)
 
        aligned_data, self.alignment_transforms, self.alignment_crop_box = alignment.align_images(
            data, groups, method=method, reference=reference, allow_rotation=allow_rotation, allow_scale=allow_scale,
            subpixel=subpixel, interpolation_order=interpolation_order, crop=crop, fill_value=fill_value, fwhm=fwhm,
            detection_sigma=detection_sigma, max_stars=max_stars, match_tolerance=match_tolerance,
            min_matches=min_matches, median_prefilter=median_prefilter, plot=plot, verbose=verbose)
 
        if variance is None and mask is None:
            return aligned_data
 
        # The variance frames and masks receive the same transformations and crop as the images:
        for name, dictionary in (("variance", variance), ("mask", mask)):
            if dictionary is not None:
                missing = [Path(file_name).name for file_name in align_list if file_name not in dictionary]
                if missing:
                    warnings.warn(f"No {name} was given for the following aligned images: " + ", ".join(missing) + ".")
        original_shape = np.shape(data[align_list[0]])
        order = interpolation_order if subpixel else 0
        aligned_variance, aligned_mask = alignment.align_variance_and_mask(variance, mask, self.alignment_transforms, self.alignment_crop_box,
                                                                           original_shape, order, crop=crop, variance_mode=variance_mode)
 
        # The calibration part of the variance (see compute_variance()) receives the same transformation, so that master() can still
        # separate it from the independent part after the alignment:
        if variance is not None and hasattr(self, "calibration_variance"):
            calibration = {name: array for name, array in self.calibration_variance.items()
                           if name in variance and np.shape(array) == np.shape(variance[name])}
            if calibration:
                aligned_calibration, _ = alignment.align_variance_and_mask(calibration, None, self.alignment_transforms,
                                                                           self.alignment_crop_box, original_shape, order, crop=crop,
                                                                           variance_mode=variance_mode)
                self.calibration_variance.update(aligned_calibration)
        return aligned_data, aligned_variance, aligned_mask