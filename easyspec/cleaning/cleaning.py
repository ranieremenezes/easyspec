import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable #this is to adjust the colorbars
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from astropy.io import fits
from scipy import stats as st
from pathlib import Path
import warnings
from scipy import odr

class cleaning:

    """This class contains all the functions necessary to perform the image reduction/cleaning, including cosmic ray removal."""

    def __init__(self,target_name=""):
        self.target_name = target_name


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
                self.bias_list = glob.glob(str(Path(bias).resolve())+'/*.fit*')
                self.all_images_list = self.all_images_list + self.bias_list
            else:
                raise TypeError("Invalid diretory for bias files.")

        if flats is not None:
            if Path(flats).is_dir():
                # Make a list with the flat images:
                self.flat_list = glob.glob(str(Path(flats).resolve())+'/*.fit*')
                self.all_images_list = self.all_images_list + self.flat_list
            else:
                raise TypeError("Invalid diretory for flat files.")

        if lamp is not None:
            if Path(lamp).is_dir():
                # Make a list with the lamp images:
                self.lamp_list = glob.glob(str(Path(lamp).resolve())+'/*.fit*')
                self.all_images_list = self.all_images_list + self.lamp_list
            else:
                raise TypeError("Invalid diretory for lamp files.")

        if standard_star is not None:
            if Path(standard_star).is_dir():
                # Make a list with the standard star images:
                self.std_list = glob.glob(str(Path(standard_star).resolve())+'/*.fit*')
                self.all_images_list = self.all_images_list + self.std_list
            else:
                raise TypeError("Invalid diretory for standard star files.")


        if targets is not None:
            if isinstance(targets,str):
                if Path(targets).is_dir():
                    # Make a list with the raw science images:
                    self.target_list = glob.glob(str(Path(targets).resolve())+'/*.fit*')
                    self.all_images_list = self.all_images_list + self.target_list
                else:
                    raise TypeError("Invalid diretory for target files.")
            elif isinstance(targets,list):
                self.target_list = []
                for target in targets:
                    if Path(targets).is_dir():
                        # Make a list with the raw science images:
                        self.target_list = self.target_list + glob.glob(str(Path(target).resolve())+'/*.fit*')
                    else:
                        raise TypeError("Invalid diretory for target files.")
                
                self.all_images_list = self.all_images_list + self.target_list
            else:
                raise RuntimeError("Variable 'target' must be a string or a list of strings.")


        if darks is not None:
            if Path(darks).is_dir():
                # Make a list with the dark files:
                self.darks_list = glob.glob(str(Path(darks).resolve())+'/*.fit*')
                self.all_images_list = self.all_images_list + self.darks_list
            else:
                raise TypeError("Invalid diretory for dark files.")

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
            Dictionary containing the path (keys) and the raw raw data for one or several images.
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
            imagenames = self.all_images_list
        elif image_type == "bias":
            imagenames = self.bias_list
        elif image_type == "flat":
            imagenames = self.flat_list
        elif image_type == "lmap":
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
    

    def master(self,Type,trimmed_data,method="median",plot=True):

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

        if method == "mode":
            if str(data_cube[0].dtype).find("int") < 0:
                method = "median"
                warnings.warn("The 'mode' method only works if the pixels have integer values. The pixels in the given file are of the "+str(data_cube[0].dtype)+
                              " type. easyspec will reset method='median'.")

        if method == "median":
            master = np.median(data_cube, axis=0)  # To combine with a median
        elif method == "mode":
            master = st.mode(data_cube, axis=0, keepdims=True)[0][0].astype(int)  # To combine with a mode
        elif method == "mean":
            master = np.mean(data_cube, axis=0)  # To combine with a mean


        # Saving master file:

        ref_hdul = fits.open(data_list[0])  # open a bias FITS file
        hdr = ref_hdul[0].header
        hdu = fits.PrimaryHDU(master) 
        hdu.header = hdr
        hdu.header['COMMENT'] = 'This is the master '+Type+' generated with easyspec. Method: '+method
        hdu.header['BZERO'] = 0  # This is to avoid a rescaling of the data
        hdul = fits.HDUList([hdu])
        hdul.writeto('master_'+Type+'.fits', overwrite=True)

        if plot:
            image_shape = np.shape(master)
            aspect_ratio = image_shape[0]/image_shape[1]
            plt.figure(figsize=(12,12*aspect_ratio)) 
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
            
            plt.show()
        
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


    def norm_master_flat(self,masterflat,degree=5,plots=True):

        """
        Function to normalize the master flat.
        
        Parameters
        ----------
        masterflat: array
            Array with master flat raw data.
        degree: int
            Polynomial degree to be fitted to the median masterflat x-profile.
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

        # Saving master flat:
        ref_flat_hdul = fits.open(self.flat_list[0])  # open a flat FITS file
        hdr = ref_flat_hdul[0].header
        hdu = fits.PrimaryHDU(normalized_master_flat)
        hdu.header = hdr
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
                else:
                    plt.title("Polynomial fit residuals")
                    plt.plot(xvals, yvals - poly_y, color='limegreen')
            plt.legend()
            
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
    

    # IMPORTANT: do a function to rotate all raw data files such that the dispersion axis is in the horizontal.
    # IMPORTANT: add function to remove reflex. E.g. a 'pad' function which takes the variable 'value' with the padding value and xy image coordinates.
    
