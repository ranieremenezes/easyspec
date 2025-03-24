Data cleaning
=============

The acquisition of photometric and spectroscopic data in astronomy is subject to several sources of noise. The major ones are as follows:

* Sensor bias – The electronic offset introduced by the detector. This is the image you have if you take a zero-second exposure with the shutter closed. This noise must be **subtracted** from the data.

* Dark current – The thermal noise present in non-refrigerated cameras. This is the image you find if you take an exposure with the shutter closed for a period of time as long as the one used to collect the target spectrum. This noise must be **subtracted** from the data.

* (non-)Flat-field – The pixel-to-pixel sensitivity variations in the light sensor. You can have an idea on how inhomogeneous is your sensor by taking a short exposure (typically less than ten seconds) of a white board homogeneously iluminated inside the telescope dome (you can also try a sky-flat). After subtracting the bias and dark frames, the 2D spectrum must be **divided** by the flat-field.

* Cosmic-ray strikes – The longer the exposure, the highest is the number of cosmic-ray strikes in your image. If you have several exposures for your target, you can get rid of them by taking the median of all your exposures (don't take the average!). If you don't have that many exposures, yo ujave to recur to algorithms that detect the sharp edges of cosmic-rray strikes and remove them.


A raw 2D spectrum, full of cosmic-ray strikes, looks like this:

.. image:: ./images/Fig_1_raw_spec.png
  :width: 700
  
  
 In the easyspec [cleaning tutorial](https://github.com/ranieremenezes/easyspec/blob/main/tutorial/Image_cleaning_easyspec.ipynb)
