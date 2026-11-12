easyspec.gemini package
=======================

Converts the raw data of the Gemini GMOS-N and GMOS-S instruments (one fits extension per amplifier, three CCDs separated by gaps,
calibrations of all kinds in the same directory) into single images, sorted into the directories expected by ``cleaning()``. See the
GMOS tutorials for examples.

The detector tables (gains, read noise, saturation and geometry of the CCDs) and the CuAr line list of the GMOS arc lamp come from the
Gemini DRAGONS pipeline (BSD 3-clause license). If you use this module, please cite DRAGONS
(`Labrie et al. 2023, RNAAS 7, 214 <https://ui.adsabs.harvard.edu/abs/2023RNAAS...7..214L>`_).

GMOS (easyspec.gemini.gmos)
---------------------------

.. automodule:: easyspec.gemini.gmos

Detector tables (easyspec.gemini.gemini_gmos_lookup)
----------------------------------------------------

Tables used by ``easyspec.gemini.gmos``.

.. automodule:: easyspec.gemini.gemini_gmos_lookup
