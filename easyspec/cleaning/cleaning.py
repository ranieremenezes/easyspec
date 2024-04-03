import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable #this is to adjust the colorbars
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from astropy.io import fits
from scipy import stats as st
from pathlib import Path
import warnings
from scipy import odr
from astropy.nddata import CCDData
from astropy import units as u
import ccdproc as ccdp
import os

class cleaning:

    """This class contains all the functions necessary to perform the image reduction/cleaning, including cosmic ray removal."""

    def __init__(self): 
        # Print the current version of easyspec       
        os.system("pip show easyspec")


    def data_paths(self,bias=None,flats=None,lamp=None,standard_star=None,targets=None,darks=None):

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

        Returns
        -------
        all_raw_data: dict
            A dictionary containing the paths to all data files (dictionary keys) and the raw data for all images (dictionary values).
        
        """
        
        self.all_images_list = []  # Master list with the paths to all data files

        if bias is not None:
            if Path(bias).is_dir():
                # Make a list with the bias files:
                self.bias_list = np.sort(glob.glob(str(Path(bias).resolve())+'/*.fit*')).tolist()
                self.all_images_list = self.all_images_list + self.bias_list
            else:
                raise TypeError("Invalid diretory for bias files.")
        else:
            self.bias_list = []

        if flats is not None:
            if Path(flats).is_dir():
                # Make a list with the flat images:
                self.flat_list = np.sort(glob.glob(str(Path(flats).resolve())+'/*.fit*')).tolist()
                self.all_images_list = self.all_images_list + self.flat_list
            else:
                raise TypeError("Invalid diretory for flat files.")
        else:
            self.flat_list = []

        if lamp is not None:
            if Path(lamp).is_dir():
                # Make a list with the lamp images:
                self.lamp_list = np.sort(glob.glob(str(Path(lamp).resolve())+'/*.fit*')).tolist()
                self.all_images_list = self.all_images_list + self.lamp_list
            else:
                raise TypeError("Invalid diretory for lamp files.")
        else:
            self.lamp_list = []

        if standard_star is not None:
            if Path(standard_star).is_dir():
                # Make a list with the standard star images:
                self.std_list = np.sort(glob.glob(str(Path(standard_star).resolve())+'/*.fit*')).tolist()
                self.all_images_list = self.all_images_list + self.std_list
            else:
                raise TypeError("Invalid diretory for standard star files.")
        else:
            self.std_list = []


        if targets is not None:
            if isinstance(targets,str):
                if Path(targets).is_dir():
                    # Make a list with the raw science images:
                    self.target_list = np.sort(glob.glob(str(Path(targets).resolve())+'/*.fit*')).tolist()
                    self.all_images_list = self.all_images_list + self.target_list
                else:
                    raise TypeError("Invalid diretory for target files.")
            elif isinstance(targets,list):
                self.target_list = []
                for target in targets:
                    if Path(target).is_dir():
                        # Make a list with the raw science images:
                        self.target_list = self.target_list + np.sort(glob.glob(str(Path(target).resolve())+'/*.fit*')).tolist()
                    else:
                        raise TypeError("Invalid diretory for target files.")
                
                self.all_images_list = self.all_images_list + self.target_list
            else:
                raise RuntimeError("Variable 'target' must be a string or a list of strings.")
        else:
            self.target_list = []


        if darks is not None:
            if Path(darks).is_dir():
                # Make a list with the dark files:
                self.darks_list = np.sort(glob.glob(str(Path(darks).resolve())+'/*.fit*')).tolist()
                self.all_images_list = self.all_images_list + self.darks_list
            else:
                raise TypeError("Invalid diretory for dark files.")
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
                print("File: ",image_name.split("/")[-1],". Shape: ",all_raw_data[image_name].shape)

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
            raise KeyError('Invalid image_type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')
        
        # Here we create an array containing the image data for all requested files:
        datacube = np.stack([datacube[image_frame] for image_frame in imagenames],axis=0)
        number_of_images = len(datacube)  # Number of individual images in the cube
        
        if specific_image is not None:
            datacube_single = []  
            datacube_single.append(datacube[specific_image])
            datacube = datacube_single
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
            single_image_name = imagenames[i].split("/")[-1].split(".")[0]  # Here we take only the image file name, i.e., we strike out all the data path and the extension.
            plt.title(single_image_name)
            ax = plt.gca()
            im = ax.imshow(np.log10(single_image_data), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            if specific_image is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.08)
                plt.colorbar(im, cax=cax,label="Counts in log scale")
            
            
        
        plt.show()
    

    def master(self,Type,trimmed_data,method="median",header_hdu_entry=0,exposure_header_entry=None,airmass_header_entry=None,plot=True):

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
            If you want to save the mean/median exposure time in the master fits file, type the header keyword for the exposure here.
        airmass_header_entry: string
            If you want to save the mean/median airmass in the master fits file, type the header keyword for the airmass here.
        plot: bool
            Keep it as True if you want to plot the master bias/dark/flat/etc.

        Returns
        -------
        master: numpy array
            The master bias raw data.
        master_TYPE.fits: data file
            This function automatically saves a file named master_TYPE.fits in the working directory.             
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
            raise KeyError('Invalid Type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')

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

        master_list = []
        for n in range(len(data_splitter[:-1])):

            if method == "median":
                master = np.median(data_cube[data_splitter[n]:data_splitter[n+1]], axis=0)  # To combine with a median
            elif method == "mode":
                master = st.mode(data_cube[data_splitter[n]:data_splitter[n+1]], axis=0, keepdims=True)[0][0].astype(int)  # To combine with a mode
            elif method == "mean":
                master = np.mean(data_cube[data_splitter[n]:data_splitter[n+1]], axis=0)  # To combine with a mean


            # Saving master file:
            hdr = fits.getheader(data_list[0],ext=header_hdu_entry)
            hdu = fits.PrimaryHDU(master,header=hdr)
            hdu.header['COMMENT'] = 'This is the master '+Type+' generated with easyspec. Method: '+method
            hdu.header['BZERO'] = 0  # This is to avoid a rescaling of the data

            # Adding exposure and airmass to the master file header:
            if Type == "target" or Type == "standard_star":
                allow_exp_and_airmass = True
            else:
                allow_exp_and_airmass = False

            if exposure_header_entry is not None and allow_exp_and_airmass:
                list_of_exposure_times = []
                for file in data_list[data_splitter[n]:data_splitter[n+1]]:
                    list_of_exposure_times.append(self.look_in_header(file,exposure_header_entry))
                
                average_exposure = np.mean(list_of_exposure_times)
                median_exposure  = np.median(list_of_exposure_times)
                summed_exposure = np.sum(list_of_exposure_times)
                hdu.header.append(('EXPSUM', summed_exposure, 'Summed exposure time for all observations'), end=True)
                hdu.header.append(('AVEXP', average_exposure, 'Average exposure time for all observations'), end=True)
                hdu.header.append(('MEDEXP', median_exposure, 'Median exposure time for all observations'), end=True)

            if airmass_header_entry is not None and allow_exp_and_airmass:
                list_of_airmasses = []
                for file in data_list[data_splitter[n]:data_splitter[n+1]]:
                    list_of_airmasses.append(self.look_in_header(file,airmass_header_entry))
                
                average_airmass = np.mean(list_of_airmasses)
                median_airmass  = np.median(list_of_airmasses)
                hdu.header.append(('AVAIRMAS', average_airmass, 'Average airmass'), end=True)
                hdu.header.append(('MDAIRMAS', median_airmass, 'Median airmass'), end=True)
            
            hdul = fits.HDUList([hdu])
            if Type == "target":
                target_name = str(Path(data_list[data_splitter[n]]).parent.resolve()).split("/")[-1]
                hdul.writeto('master_'+Type+f'_{target_name}.fits', overwrite=True)
            else:
                hdul.writeto('master_'+Type+'.fits', overwrite=True)

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
                    vmax = np.median(np.log10(master))
                    im = ax.imshow(np.log10(master), origin='lower', cmap='gray', vmax=vmax)
                else:
                    im = ax.imshow(np.log10(master), origin='lower', cmap='gray')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.08)
                plt.colorbar(im, cax=cax,label="Counts in log scale")
            
            if n == (len(data_splitter[:-1]) - 1):
                plt.show()
        
        if len(data_splitter) > 2:
            return master_list
        else:
            return master

    def debias(self, trimmed_data, masterbias, Type="all", pad_with_zeros=True):

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


    def sub_dark(self, debiased_data, masterdark, Type="all", pad_with_zeros=True):

        """
        Function to subtract the masterdark from the rest of the data.
        
        Parameters
        ----------
        debiased_data: dict
            Dictionary containing debiased data. This dictionary can even contain the raw dark files, but should not contain the raw bias files. 
            At the end we select only the entry given in the variable Type.
        masterdark: array
            Array with master dark raw data.
        Type: string
            Type of data files to subtract the master bias. Options are: "all", "flat", "lamp", "standard_star", and "target".
        pad_with_zeros: bool
            If True, easyspec will look for negative-value pixels after the dark subtraction and set them to zero. 

        Returns
        -------
        subdark_data_out: dict
            Dictionary containing the data corrected for dark.
        """

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
            else:
                raise KeyError('There are two possibilities for this error:\n1) There is an invalid value for the variable Type. Options are: "all", "flat", "lamp", "standard_star", and "target".'+
                               '\n2)If the value of Type is fine, please be sure that you did not change the keys in the debiased_data dictionary.')

        subdark_data_out = {}  # Dictionary for the sub_dark images

        for i in range(len(subdark_list)):  
            subdark_data_out[subdark_list[i]] = debiased_data[subdark_list[i]] - masterdark  # Subtracting the master dark from each one of the raw images
            a,b=np.where(subdark_data_out[subdark_list[i]]<=0)
            if len(a) > 0 and pad_with_zeros:
                subdark_data_out[subdark_list[i]][a,b] = 0  # Padding negative pixels with zeros
        
        return subdark_data_out


    def norm_master_flat(self,masterflat,degree=5,header_hdu_entry=0,plots=True):

        """
        Function to normalize the master flat.
        
        Parameters
        ----------
        masterflat: array
            Array with master flat raw data.
        degree: int
            Polynomial degree to be fitted to the median masterflat x-profile.
        header_hdu_entry: int
            HDU extension where we can find the header. Please check your data before choosing this value.
        plots: bool
            If True, easyspec will plot the master flat profile, fit residuals and normalized master flat. 

        Returns
        -------
        normalized_master_flat: array
            Normalized master flat raw data.
        norm_master_flat.fits: data file
            This function automatically saves a file named "norm_master_flat.fits" in the working directory.
        """


        yvals = np.median(masterflat, axis=0)
        xvals = np.arange(np.shape(masterflat)[1])

        poly_model = odr.polynomial(degree)  # Using a polynomial model of order 'degree'
        data = odr.Data(xvals, yvals)
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # Running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y = poly(xvals)

        fit_matrix = np.ones(np.shape(masterflat))*poly_y
        normalized_master_flat = masterflat/fit_matrix

        # Saving norm master flat:
        hdr = fits.getheader(self.flat_list[0],ext=header_hdu_entry)
        hdu = fits.PrimaryHDU(normalized_master_flat,header=hdr)
        hdu.header['COMMENT'] = 'This is the normalized master flat generated with easyspec'
        hdu.header['BZERO'] = 0 #This is to avoid a rescaling of the data
        hdul = fits.HDUList([hdu])
        hdul.writeto('norm_master_flat.fits', overwrite=True)

        if plots:
            Matrix_shape = gridspec.GridSpec(1, 2)  # Define the image grid
            plt.figure(figsize=(12,4))  # Set the figure size
            for i in range(2): 
                plt.subplot(Matrix_shape[i])
                plt.grid(which="both",linestyle=":")
                if i == 0:
                    plt.title("Master flat profile")
                    plt.plot(xvals, yvals, 'x', alpha=0.5,label="column-median data")
                    plt.plot(xvals, poly_y, color='limegreen',label="ODR polynomial fit")
                    plt.legend()
                else:
                    plt.title("Polynomial fit residuals")
                    plt.plot(xvals, yvals - poly_y, color='limegreen')
            
            
            plt.figure()
            image_shape = np.shape(normalized_master_flat)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio))
            ax = plt.gca()
            plt.title('Normalized Master Flat')
            im = ax.imshow(np.log10(normalized_master_flat), origin='lower', cmap='gray')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.08)
            plt.colorbar(im, cax=cax,label="Norm. counts in log scale")

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


    def pad(self,image_data,value,x1,x2,y1,y2,plot=True):

        """
        We use this function to pad a specific section of an image with a given value. It is particularly useful when handling the master files.
        
        Parameters
        ----------
        image_data: array
            Array containing the raw data for one image.
        value: float
            The value that is going to be substituted in the image section defided by the limits x1,x2,y1,y2.
        x1,x2,y1,y2: integers
            Values giving the image section limits for padding.
        plot: bool
            If True, easyspec will print the padded image.

        Returns
        -------
        image_data: array
            An array with the new padded image.
        """

        image_data = np.copy(image_data)

        image_data[y1:y2,x1:x2] = value
        if plot:
            image_shape = np.shape(image_data)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio)) 
            plt.title('Padded image')
            ax = plt.gca()
            im = ax.imshow(np.log10(image_data), origin='lower', cmap='gray')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.08)
            plt.colorbar(im, cax=cax,label="Counts in log scale")
            plt.show()
        
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
        im = ax.imshow(np.log10(image_data), origin='lower', cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.08)
        plt.colorbar(im, cax=cax,label="Counts in log scale")

        if save:
            plt.savefig(figure_name+"."+format,bbox_inches='tight')
        
        plt.show()


    def flatten(self,debiased_data,norm_master_flat,Type="all"):

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

        Returns
        -------
        flattened_debiased_data_out: dict
            Dictionary containing the flattened data.
        """

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

        for i in range(len(flatten_list)):  
            flattened_debiased_data_out[flatten_list[i]] = debiased_data[flatten_list[i]]/norm_master_flat  # Flattening the images with the normalized master flat.
        
        return flattened_debiased_data_out
    

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

    def CR_and_gain_corrections(self,flattened_debiased_data,gain=None,gain_header_entry="GAIN",readnoise=None,readnoise_header_entry="RDNOISE",Type="all",sigclip=5):

        """
        Function to remove cosmic rays from target and standard star raw data based on the LACosmic method originally developed and implemented for IRAF by Pieter G. van Dokkum.
        This function also renormalizes the data according to the CCD/CMOS gain. We always need to work in electrons for cosmic ray detection.

        Parameters
        ----------
        flattened_debiased_data: dict
            A dictionary containing the data that needs correction.
        gain: float
            This is the CCD gain. We will use it to convert the image units from ADU to electrons. If gain = None, easyspec will look for this entry automatically in the file header.
        gain_header_entry: string
            We use this only if gain=None. This must be the keyword containing the gain value in the fits file header.
        readnoise: float
            This is the CCD read noise, used to generate the noise model of the image. If readnoise = None, easyspec will look for this entry automatically in the file header.
        readnoise_header_entry: string
            We use this only if readnoise=None. This must be the keyword containing the read noise value in the fits file header.
        Type: string
            Type of data files to apply the corrections. Options are: "all", "lamp", "standard_star", and "target".
        sigclip: float
            Laplacian-to-noise limit for cosmic ray detection. Lower values will flag more pixels as cosmic rays.
            
        Returns
        -------
        CR_corrected_data: dict
            Dictionary containing the data corrected for gain and cosmic rays.
        
        """

        rm_cr_list = []
        for file_name in flattened_debiased_data.keys():
            if Type == "all" and file_name not in self.bias_list and file_name not in self.darks_list and file_name not in self.flat_list:
                rm_cr_list = rm_cr_list + [file_name]
            elif Type == "target" and file_name in self.target_list:
                rm_cr_list = rm_cr_list + [file_name]
            elif Type == "standard_star" and file_name in self.std_list:
                rm_cr_list = rm_cr_list + [file_name]
            elif Type == "lamp" and file_name in self.lamp_list:
                rm_cr_list = rm_cr_list + [file_name]
            
        if len(rm_cr_list) == 0:
            raise KeyError('Empty list with target and standard star data. There are two possibilities for this error:\n1) There is an invalid value for the variable Type. '
                           +'Options are: "all", "standard_star", and "target".'+
                            '\n2)If the value of Type is fine, please be sure that you did not change the keys in the input dictionary.')
 

        CR_corrected_data = {}
        for file_name in rm_cr_list:
            if readnoise is None:
                readnoise = self.look_in_header(file_name,readnoise_header_entry)
            if gain is None:
                gain = self.look_in_header(file_name,gain_header_entry)
            
            CR_corrected_data[file_name] = CCDData(flattened_debiased_data[file_name], unit='adu')  # The data is now in the CCDData format. We make it get better to a numpy array below.
            CR_corrected_data[file_name] = ccdp.gain_correct(CR_corrected_data[file_name], gain * u.electron / u.adu)
            CR_corrected_data[file_name] = ccdp.cosmicray_lacosmic(CR_corrected_data[file_name], readnoise=readnoise, sigclip=sigclip, verbose=True)

        # The mask saved in CR_corrected_data includes cosmic rays identified by the function cosmicray_lacosmic.
        # The sum of the mask indicates how many pixels have been identified as cosmic rays.
        # Let's apply the cosmic-ray masks:

        for file_name in rm_cr_list:
            q = CR_corrected_data[file_name].mask
            q[CR_corrected_data[file_name].mask]=False
            CR_corrected_data[file_name].mask.sum()
            CR_corrected_data[file_name] = np.asarray(CR_corrected_data[file_name])  # Back to the numpy array format.


        return CR_corrected_data


    def save_fits_files(self,datacube,output_directory,header_hdu_entry=0,Type="all"):

        """
        This function saves the raw data contained in the dictionary 'datacube' as fits files in the directory 'output_directory'.
        
        Parameters
        ----------
        datacube: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data for their respective images (dictionary values).
        output_directory: string
            String with the path to the output directory.
        header_hdu_entry: int
            HDU extension where we can find the header. Please check your data before choosing this value.
        Type: string
            Type of data files to be saved. Options are: "all", "bias", "flat", "lamp", "standard_star", "target", and "dark".

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
            raise KeyError('Invalid variable Type. Options are: "bias", "flat", "lamp", "standard_star", "target", and "dark".')
        
        for n,file_name in enumerate(imagenames):
            hdr = fits.getheader(file_name,ext=header_hdu_entry)
            hdu = fits.PrimaryHDU(datacube[file_name],header=hdr)
            hdu.header['COMMENT'] = 'File generated with easyspec'
            hdu.header['BZERO'] = 0  # This is to avoid a rescaling of the data
            hdul = fits.HDUList([hdu])
            hdul.writeto(str(Path(output_directory).resolve())+'/'+Type+'_{:03d}'.format(n)+'.fits', overwrite=True)


    def vertical_align(self, CR_corrected_data, Type="all"):

        """
        This function is used to align target and standard star data taht are shifted in the vertical axis.
        Although this function does not align lamp images, it accepts the raw lamp data in the dictionary CR_corrected_data.
        In the last step of this function, all data will be trimmed according to the alignment cuts, including the lamp data (if provided). 
        
        Parameters
        ----------
        CR_corrected_data: dict
            A dictionary containing the paths to the data files (dictionary keys) and the data for their respective images (dictionary values).
        Type: string
            Type of data files to align. Options are: "all", "standard_star", and "target".

        Returns
        -------
        aligned_data: dict
            Dictionary containing the aligned and trimmed data passed in the variable CR_corrected_data.
        
        """

        CR_corrected_data = CR_corrected_data.copy()

        align_list = []
        for file_name in CR_corrected_data.keys():
            if Type == "all" and file_name not in self.bias_list and file_name not in self.darks_list and file_name not in self.flat_list:
                align_list = align_list + [file_name]
            elif Type == "target" and file_name in self.target_list:
                align_list = align_list + [file_name]
            elif Type == "standard_star" and file_name in self.std_list:
                align_list = align_list + [file_name]


        if Type == "all" or Type == "target":
            data_path = ""
            data_splitter = []
            for n,file in enumerate(align_list):
                parent_directory = str(Path(file).parent.resolve())
                if parent_directory != data_path:
                    data_path = parent_directory
                    data_splitter.append(n)
            data_splitter.append(len(align_list))
        else:
            data_splitter = [0,len(align_list)]



        maximum_y1_cut = 0
        maximum_y2_cut = 0
        for n in range(len(data_splitter[:-1])): 
            reference_image = align_list[data_splitter[n]]  # Selecting the first image as the reference image
            print("Reference image: ", reference_image)

            # Selecting the strongest horizontal and vertical lines:
            ref_yvals = np.argmax(CR_corrected_data[reference_image], axis=0)  # Selects the index of the pixel with highest value in each column     

            for file_name in align_list[data_splitter[n]:data_splitter[n+1]]:
                if file_name == reference_image:
                    continue

                print("Current image: ", file_name)
                yvals = np.argmax(CR_corrected_data[file_name], axis=0)  # Selects the index of the pixel with highest value in each column

                difference_y = ref_yvals - yvals
                difference_y = int(st.mode(difference_y, keepdims=True)[0][0])  # To find the global y shift

                print("Vertical shift (pixels): ", difference_y)

                # Vertical alignment: 
                if difference_y < 0:
                    # Remember that y increases from the top to the bottom!!!
                    temporary_image = np.concatenate([CR_corrected_data[file_name], np.zeros([difference_y,np.shape(CR_corrected_data[file_name])[1]])])
                    temporary_image = self.trim({'temporary' : temporary_image},x1=0,x2=np.shape(CR_corrected_data[file_name])[1],y1=difference_y,y2=np.shape(CR_corrected_data[file_name])[0]+difference_y)
                    CR_corrected_data[file_name] = temporary_image['temporary']  # Remember that the function trim above returns a dictionary.
                    if difference_y < maximum_y2_cut:
                        maximum_y2_cut = difference_y

                elif difference_y > 0:
                    temporary_image = np.concatenate([np.zeros([difference_y,np.shape(CR_corrected_data[file_name])[1]]), CR_corrected_data[file_name]])
                    temporary_image = self.trim({'temporary' : temporary_image},x1=0,x2=np.shape(CR_corrected_data[file_name])[1],y1=0,y2=np.shape(CR_corrected_data[file_name])[0])
                    CR_corrected_data[file_name] = temporary_image['temporary']  # Remember that the function trim above returns a dictionary.
                    if difference_y > maximum_y1_cut:
                        maximum_y1_cut = difference_y
                else:
                    print("Images are already aligned in the vertical.")

        print("Final vertical cuts (y1, y2): ",maximum_y1_cut, maximum_y2_cut)
        aligned_data = self.trim(CR_corrected_data,x1=0,x2=np.shape(CR_corrected_data[file_name])[1],y1=maximum_y1_cut,y2=np.shape(CR_corrected_data[file_name])[0]+maximum_y2_cut)  # Remember that maximum_y2_cut is negative

            
        return aligned_data
    
    