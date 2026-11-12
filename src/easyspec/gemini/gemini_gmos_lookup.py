"""
easyspec gemini_gmos_lookup module.

Detector tables of the Gemini GMOS-N and GMOS-S instruments, used by easyspec.gemini_gmos: the gain and the read noise of each
amplifier (which depend on the date, the read-out speed and the gain setting), the full-well depths, the gaps and the relative positions
of the three CCDs, and the pixel scales.

These tables were taken from DRAGONS, the data reduction software of the Gemini Observatory (gemini_instruments/gmos/lookup.py and
geminidr/gmos/lookups/geometry_conf.py; https://github.com/GeminiDRSoftware/DRAGONS; Labrie et al. 2023, RNAAS 7, 214), which is
distributed under the BSD 3-Clause license:

    Copyright (C) 2014-2026 Association of Universities for Research in Astronomy, Inc. (AURA)

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following
    conditions are met: (1) redistributions of source code must retain the above copyright notice, this list of conditions and the
    following disclaimer; (2) redistributions in binary form must reproduce the above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or other materials provided with the distribution; (3) neither the name of
    the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
    NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


# Gain (electrons/ADU) and read noise (electrons) of each amplifier. One table per epoch: (first UT date of validity, table). The
# tables are searched from the most recent to the oldest one. In each table: {amplifier name (AMPNAME keyword): {(read-out speed,
# gain setting): (gain, read noise)}}, with the read-out speed "slow" or "fast" and the gain setting "low" or "high".
AMPLIFIER_EPOCHS = [
    ("2023-12-14", {
        'BI13-20-4k-1, 1': {('slow', 'low'): (1.568, 3.99), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.04, 8.17), ('fast', 'low'): (1.91, 5.88)},
        'BI13-20-4k-1, 2': {('slow', 'low'): (1.62, 4.12), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.05, 8.75), ('fast', 'low'): (1.97, 6.01)},
        'BI13-20-4k-1, 3': {('slow', 'low'): (1.618, 4.12), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.18, 8.94), ('fast', 'low'): (1.95, 6.88)},
        'BI13-20-4k-1, 4': {('slow', 'low'): (1.675, 4.06), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.21, 9.9), ('fast', 'low'): (2.01, 8.02)},
        'BI12-09-4k-2, 1': {('slow', 'low'): (1.664, 4.2), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.29, 8.84), ('fast', 'low'): (2.01, 5.86)},
        'BI12-09-4k-2, 2': {('slow', 'low'): (1.633, 3.88), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.14, 8.19), ('fast', 'low'): (1.96, 5.39)},
        'BI12-09-4k-2, 3': {('slow', 'low'): (1.65, 3.98), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.13, 8.91), ('fast', 'low'): (1.97, 6.34)},
        'BI12-09-4k-2, 4': {('slow', 'low'): (1.69, 4.2), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.29, 9.42), ('fast', 'low'): (2.03, 6.71)},
        'BI13-18-4k-2, 1': {('slow', 'low'): (1.654, 4.55), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.09, 8.8), ('fast', 'low'): (1.95, 6.79)},
        'BI13-18-4k-2, 2': {('slow', 'low'): (1.587, 4.02), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.86, 8.03), ('fast', 'low'): (1.89, 5.98)},
        'BI13-18-4k-2, 3': {('slow', 'low'): (1.63, 4.35), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.1, 8.53), ('fast', 'low'): (1.95, 6.13)},
        'BI13-18-4k-2, 4': {('slow', 'low'): (1.604, 4.04), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.03, 8.13), ('fast', 'low'): (1.95, 6.1)},
        'BI11-41-4k-2, 1': {('slow', 'low'): (2.038, 4.22), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.758, 11.74), ('fast', 'low'): (1.719, 7.01)},
        'BI11-41-4k-2, 2': {('slow', 'low'): (2.016, 4.29), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.708, 10.19), ('fast', 'low'): (1.711, 6.93)},
        'BI11-41-4k-2, 3': {('slow', 'low'): (2.02, 4.2), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.719, 10.45), ('fast', 'low'): (1.715, 6.37)},
        'BI11-41-4k-2, 4': {('slow', 'low'): (1.943, 4.05), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.518, 9.06), ('fast', 'low'): (1.652, 5.19)},
        'BI13-19-4k-3, 1': {('slow', 'low'): (1.798, 3.79), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.091, 11.27), ('fast', 'low'): (1.533, 5.2)},
        'BI13-19-4k-3, 2': {('slow', 'low'): (1.745, 3.68), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.956, 9.46), ('fast', 'low'): (1.498, 5.82)},
        'BI13-19-4k-3, 3': {('slow', 'low'): (1.787, 3.59), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.082, 7.99), ('fast', 'low'): (1.525, 5.24)},
        'BI13-19-4k-3, 4': {('slow', 'low'): (1.787, 4.22), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.1, 9.52), ('fast', 'low'): (1.531, 5.46)},
        'BI12-34-4k-1, 1': {('slow', 'low'): (1.594, 3.39), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.581, 8.85), ('fast', 'low'): (1.445, 4.66)},
        'BI12-34-4k-1, 2': {('slow', 'low'): (1.699, 3.67), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.893, 7.96), ('fast', 'low'): (1.545, 5.1)},
        'BI12-34-4k-1, 3': {('slow', 'low'): (1.681, 3.39), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.825, 8.48), ('fast', 'low'): (1.478, 5.39)},
        'BI12-34-4k-1, 4': {('slow', 'low'): (1.77, 3.53), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.082, 7.83), ('fast', 'low'): (1.549, 9.57)},
    }),
    ("2017-02-24", {
        'BI13-20-4k-1, 1': {('slow', 'low'): (1.568, 3.99), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.04, 8.17), ('fast', 'low'): (1.91, 5.88)},
        'BI13-20-4k-1, 2': {('slow', 'low'): (1.62, 4.12), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.05, 8.75), ('fast', 'low'): (1.97, 6.01)},
        'BI13-20-4k-1, 3': {('slow', 'low'): (1.618, 4.12), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.18, 8.94), ('fast', 'low'): (1.95, 6.88)},
        'BI13-20-4k-1, 4': {('slow', 'low'): (1.675, 4.06), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.21, 9.9), ('fast', 'low'): (2.01, 8.02)},
        'BI12-09-4k-2, 1': {('slow', 'low'): (1.664, 4.2), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.29, 8.84), ('fast', 'low'): (2.01, 5.86)},
        'BI12-09-4k-2, 2': {('slow', 'low'): (1.633, 3.88), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.14, 8.19), ('fast', 'low'): (1.96, 5.39)},
        'BI12-09-4k-2, 3': {('slow', 'low'): (1.65, 3.98), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.13, 8.91), ('fast', 'low'): (1.97, 6.34)},
        'BI12-09-4k-2, 4': {('slow', 'low'): (1.69, 4.2), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.29, 9.42), ('fast', 'low'): (2.03, 6.71)},
        'BI13-18-4k-2, 1': {('slow', 'low'): (1.654, 4.55), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.09, 8.8), ('fast', 'low'): (1.95, 6.79)},
        'BI13-18-4k-2, 2': {('slow', 'low'): (1.587, 4.02), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.86, 8.03), ('fast', 'low'): (1.89, 5.98)},
        'BI13-18-4k-2, 3': {('slow', 'low'): (1.63, 4.35), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.1, 8.53), ('fast', 'low'): (1.95, 6.13)},
        'BI13-18-4k-2, 4': {('slow', 'low'): (1.604, 4.04), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.03, 8.13), ('fast', 'low'): (1.95, 6.1)},
        'BI5-36-4k-2, 1': {('slow', 'low'): (1.852, 4.21), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.24, 8.93), ('fast', 'low'): (1.566, 6.68)},
        'BI5-36-4k-2, 2': {('slow', 'low'): (1.878, 4.24), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.33, 9.13), ('fast', 'low'): (1.592, 7.1)},
        'BI5-36-4k-2, 3': {('slow', 'low'): (1.874, 4.28), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.3, 8.85), ('fast', 'low'): (1.593, 6.35)},
        'BI5-36-4k-2, 4': {('slow', 'low'): (1.834, 4.04), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.183, 8.07), ('fast', 'low'): (1.556, 5.49)},
        'BI11-33-4k-1, 1': {('slow', 'low'): (1.878, 4.29), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.334, 8.35), ('fast', 'low'): (1.613, 5.19)},
        'BI11-33-4k-1, 2': {('slow', 'low'): (1.84, 4.07), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.225, 8.59), ('fast', 'low'): (1.582, 6.14)},
        'BI11-33-4k-1, 3': {('slow', 'low'): (1.933, 4.24), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.459, 8.62), ('fast', 'low'): (1.648, 7.38)},
        'BI11-33-4k-1, 4': {('slow', 'low'): (1.908, 3.9), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.407, 8.02), ('fast', 'low'): (1.634, 5.77)},
        'BI12-34-4k-1, 1': {('slow', 'low'): (1.652, 3.53), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.828, 7.94), ('fast', 'low'): (1.445, 6.36)},
        'BI12-34-4k-1, 2': {('slow', 'low'): (1.761, 3.72), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.129, 8.3), ('fast', 'low'): (1.545, 6.8)},
        'BI12-34-4k-1, 3': {('slow', 'low'): (1.724, 3.61), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.894, 7.99), ('fast', 'low'): (1.478, 7.88)},
        'BI12-34-4k-1, 4': {('slow', 'low'): (1.813, 4.15), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.12, 9.34), ('fast', 'low'): (1.549, 7.72)},
    }),
    ("2015-08-26", {
        'e2v 10031-23-05, right': {('slow', 'low'): (2.31, 3.17), ('slow', 'high'): (4.52, 3.2), ('fast', 'high'): (5.34, 7.8), ('fast', 'low'): (2.53, 4.2)},
        'e2v 10031-01-03, right': {('slow', 'low'): (2.27, 3.22), ('slow', 'high'): (4.39, 1.5), ('fast', 'high'): (5.27, 5.8), ('fast', 'low'): (2.5, 4.1)},
        'e2v 10031-18-04, left': {('slow', 'low'): (2.17, 3.46), ('slow', 'high'): (4.82, 2.8), ('fast', 'high'): (5.16, 6.4), ('fast', 'low'): (2.43, 4.3)},
        'e2v 10031-23-05, left': {('slow', 'low'): (2.31, 3.41), ('slow', 'high'): (4.59, 3.1), ('fast', 'high'): (5.36, 7.0), ('fast', 'low'): (2.53, 4.3)},
        'e2v 10031-01-03, left': {('slow', 'low'): (2.21, 3.2), ('slow', 'high'): (4.59, 2.4), ('fast', 'high'): (5.1, 5.7), ('fast', 'low'): (2.41, 4.1)},
        'e2v 10031-18-04, right': {('slow', 'low'): (2.33, 3.44), ('slow', 'high'): (4.33, 3.1), ('fast', 'high'): (5.41, 6.3), ('fast', 'low'): (2.56, 4.4)},
        'BI5-36-4k-2, 1': {('slow', 'low'): (1.852, 4.21), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.24, 8.93), ('fast', 'low'): (1.566, 6.68)},
        'BI5-36-4k-2, 2': {('slow', 'low'): (1.878, 4.24), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.33, 9.13), ('fast', 'low'): (1.592, 7.1)},
        'BI5-36-4k-2, 3': {('slow', 'low'): (1.874, 4.28), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.3, 8.85), ('fast', 'low'): (1.593, 6.35)},
        'BI5-36-4k-2, 4': {('slow', 'low'): (1.834, 4.04), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.183, 8.07), ('fast', 'low'): (1.556, 5.49)},
        'BI11-33-4k-1, 1': {('slow', 'low'): (1.878, 4.29), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.334, 8.35), ('fast', 'low'): (1.613, 5.19)},
        'BI11-33-4k-1, 2': {('slow', 'low'): (1.84, 4.07), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.225, 8.59), ('fast', 'low'): (1.582, 6.14)},
        'BI11-33-4k-1, 3': {('slow', 'low'): (1.933, 4.24), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.459, 8.62), ('fast', 'low'): (1.648, 7.38)},
        'BI11-33-4k-1, 4': {('slow', 'low'): (1.908, 3.9), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.407, 8.02), ('fast', 'low'): (1.634, 5.77)},
        'BI12-34-4k-1, 1': {('slow', 'low'): (1.652, 3.53), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.828, 7.94), ('fast', 'low'): (1.445, 6.36)},
        'BI12-34-4k-1, 2': {('slow', 'low'): (1.761, 3.72), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.129, 8.3), ('fast', 'low'): (1.545, 6.8)},
        'BI12-34-4k-1, 3': {('slow', 'low'): (1.724, 3.61), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.894, 7.99), ('fast', 'low'): (1.478, 7.88)},
        'BI12-34-4k-1, 4': {('slow', 'low'): (1.813, 4.15), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.12, 9.34), ('fast', 'low'): (1.549, 7.72)},
    }),
    ("2006-08-31", {
        'e2v 10031-23-05, right': {('slow', 'low'): (2.31, 3.17), ('slow', 'high'): (4.52, 3.2), ('fast', 'high'): (5.34, 7.8), ('fast', 'low'): (2.53, 4.2)},
        'e2v 10031-01-03, right': {('slow', 'low'): (2.27, 3.22), ('slow', 'high'): (4.39, 1.5), ('fast', 'high'): (5.27, 5.8), ('fast', 'low'): (2.5, 4.1)},
        'e2v 10031-18-04, left': {('slow', 'low'): (2.17, 3.46), ('slow', 'high'): (4.82, 2.8), ('fast', 'high'): (5.16, 6.4), ('fast', 'low'): (2.43, 4.3)},
        'e2v 10031-23-05, left': {('slow', 'low'): (2.31, 3.41), ('slow', 'high'): (4.59, 3.1), ('fast', 'high'): (5.36, 7.0), ('fast', 'low'): (2.53, 4.3)},
        'e2v 10031-01-03, left': {('slow', 'low'): (2.21, 3.2), ('slow', 'high'): (4.59, 2.4), ('fast', 'high'): (5.1, 5.7), ('fast', 'low'): (2.41, 4.1)},
        'e2v 10031-18-04, right': {('slow', 'low'): (2.33, 3.44), ('slow', 'high'): (4.33, 3.1), ('fast', 'high'): (5.41, 6.3), ('fast', 'low'): (2.56, 4.4)},
        'EEV 9273-16-03, right': {('slow', 'low'): (2.1, 3.5), ('slow', 'high'): (4.405, 5.1), ('fast', 'high'): (4.839, 6.6), ('fast', 'low'): (2.302, 4.9)},
        'EEV 9273-20-04, right': {('slow', 'low'): (2.337, 3.3), ('slow', 'high'): (4.916, 5.1), ('fast', 'high'): (5.432, 7.0), ('fast', 'low'): (2.578, 4.7)},
        'EEV 9273-20-03, left': {('slow', 'low'): (2.3, 3.0), ('slow', 'high'): (4.841, 4.8), ('fast', 'high'): (5.397, 6.9), ('fast', 'low'): (2.56, 4.8)},
        'EEV 9273-16-03, left': {('slow', 'low'): (2.012, 3.5), ('slow', 'high'): (4.247, 4.6), ('fast', 'high'): (4.629, 5.9), ('fast', 'low'): (2.189, 4.5)},
        'EEV 9273-20-04, left': {('slow', 'low'): (2.28, 3.9), ('slow', 'high'): (4.809, 4.8), ('fast', 'high'): (5.275, 11.0), ('fast', 'low'): (2.498, 4.8)},
        'EEV 9273-20-03, right': {('slow', 'low'): (2.257, 3.3), ('slow', 'high'): (4.779, 5.0), ('fast', 'high'): (5.354, 8.0), ('fast', 'low'): (2.526, 4.7)},
        'EEV 8056-20-03, left': {('slow', 'low'): (1.94, 3.2), ('slow', 'high'): (4.1, 4.2), ('fast', 'high'): (5.3, 6.8), ('fast', 'low'): (2.4, 4.4)},
        'EEV 8194-19-04, left': {('slow', 'low'): (2.076, 3.85), ('slow', 'high'): (4.532, 4.81), ('fast', 'high'): (5.051, 7.35), ('fast', 'low'): (2.295, 4.42)},
        'EEV 8261-07-04, right': {('slow', 'low'): (2.097, 3.16), ('slow', 'high'): (4.411, 4.34), ('fast', 'high'): (4.833, 7.88), ('fast', 'low'): (2.26, 4.09)},
        'EEV 8056-20-03, right': {('slow', 'low'): (2.0, 3.4), ('slow', 'high'): (4.2, 4.9), ('fast', 'high'): (5.3, 6.3), ('fast', 'low'): (2.4, 4.3)},
        'EEV 8194-19-04, right': {('slow', 'low'): (2.131, 3.83), ('slow', 'high'): (4.592, 4.96), ('fast', 'high'): (4.95, 11.37), ('fast', 'low'): (2.288, 5.03)},
        'EEV 8261-07-04, left': {('slow', 'low'): (2.056, 3.27), ('slow', 'high'): (4.381, 4.81), ('fast', 'high'): (4.868, 8.56), ('fast', 'low'): (2.264, 4.63)},
        'EEV 2037-06-03, left': {('slow', 'low'): (2.372, 3.98), ('slow', 'high'): (4.954, 5.7), ('fast', 'high'): (5.054, 6.84), ('fast', 'low'): (2.408, 4.49)},
        'EEV 2037-06-03, right': {('slow', 'low'): (2.403, 4.01), ('slow', 'high'): (4.862, 5.19), ('fast', 'high'): (5.253, 6.54), ('fast', 'low'): (2.551, 5.07)},
        'BI5-36-4k-2, 1': {('slow', 'low'): (1.652, 4.24), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.857, 10.426), ('fast', 'low'): (1.44, 7.725)},
        'BI5-36-4k-2, 2': {('slow', 'low'): (1.72, 4.0), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (5.062, 8.109), ('fast', 'low'): (1.495, 6.891)},
        'BI5-36-4k-2, 3': {('slow', 'low'): (1.7, 4.25), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.849, 8.248), ('fast', 'low'): (1.51, 7.244)},
        'BI5-36-4k-2, 4': {('slow', 'low'): (1.626, 4.03), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.688, 8.283), ('fast', 'low'): (1.437, 7.114)},
        'BI11-33-4k-1, 1': {('slow', 'low'): (1.664, 4.12), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.867, 8.223), ('fast', 'low'): (1.422, 5.986)},
        'BI11-33-4k-1, 2': {('slow', 'low'): (1.691, 3.83), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.706, 7.939), ('fast', 'low'): (1.409, 6.66)},
        'BI11-33-4k-1, 3': {('slow', 'low'): (1.673, 3.98), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.902, 7.65), ('fast', 'low'): (1.333, 5.658)},
        'BI11-33-4k-1, 4': {('slow', 'low'): (1.739, 3.8), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.938, 8.255), ('fast', 'low'): (1.561, 7.49)},
        'BI12-34-4k-1, 1': {('slow', 'low'): (1.519, 3.46), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.343, 7.46), ('fast', 'low'): (1.304, 6.556)},
        'BI12-34-4k-1, 2': {('slow', 'low'): (1.51, 3.35), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.334, 6.522), ('fast', 'low'): (1.304, 5.401)},
        'BI12-34-4k-1, 3': {('slow', 'low'): (1.51, 3.25), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.384, 7.383), ('fast', 'low'): (1.336, 5.907)},
        'BI12-34-4k-1, 4': {('slow', 'low'): (1.613, 3.5), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.588, 6.574), ('fast', 'low'): (1.424, 5.446)},
        'BI12-34-4k-1 1': {('slow', 'low'): (1.519, 3.46), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.343, 7.46), ('fast', 'low'): (1.304, 6.556)},
        'BI12-34-4k-1 2': {('slow', 'low'): (1.51, 3.35), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.334, 6.522), ('fast', 'low'): (1.304, 5.401)},
        'BI12-34-4k-1 3': {('slow', 'low'): (1.51, 3.25), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.384, 7.383), ('fast', 'low'): (1.336, 5.907)},
        'BI12-34-4k-1 4': {('slow', 'low'): (1.613, 3.5), ('slow', 'high'): (1.0, 0.0), ('fast', 'high'): (4.588, 6.574), ('fast', 'low'): (1.424, 5.446)},
    }),
    ("0001-01-01", {
        'EEV 9273-16-03, right': {('slow', 'low'): (2.1, 3.5), ('slow', 'high'): (4.405, 5.1), ('fast', 'high'): (4.839, 6.6), ('fast', 'low'): (2.302, 4.9)},
        'EEV 9273-20-04, right': {('slow', 'low'): (2.337, 3.3), ('slow', 'high'): (4.916, 5.1), ('fast', 'high'): (5.432, 7.0), ('fast', 'low'): (2.578, 4.7)},
        'EEV 9273-20-03, left': {('slow', 'low'): (2.3, 3.0), ('slow', 'high'): (4.841, 4.8), ('fast', 'high'): (5.397, 6.9), ('fast', 'low'): (2.56, 4.8)},
        'EEV 9273-16-03, left': {('slow', 'low'): (2.012, 3.5), ('slow', 'high'): (4.247, 4.6), ('fast', 'high'): (4.629, 5.9), ('fast', 'low'): (2.189, 4.5)},
        'EEV 9273-20-04, left': {('slow', 'low'): (2.28, 3.9), ('slow', 'high'): (4.809, 4.8), ('fast', 'high'): (5.275, 11.0), ('fast', 'low'): (2.498, 4.8)},
        'EEV 9273-20-03, right': {('slow', 'low'): (2.257, 3.3), ('slow', 'high'): (4.779, 5.0), ('fast', 'high'): (5.354, 8.0), ('fast', 'low'): (2.526, 4.7)},
        'EEV 8056-20-03, left': {('slow', 'low'): (1.94, 3.2), ('slow', 'high'): (4.1, 4.2), ('fast', 'high'): (5.3, 6.8), ('fast', 'low'): (2.4, 4.4)},
        'EEV 8194-19-04, left': {('slow', 'low'): (2.076, 3.69), ('slow', 'high'): (4.532, 4.69), ('fast', 'high'): (5.051, 6.14), ('fast', 'low'): (2.295, 4.73)},
        'EEV 8261-07-04, right': {('slow', 'low'): (2.097, 3.31), ('slow', 'high'): (4.411, 4.45), ('fast', 'high'): (4.833, 6.2), ('fast', 'low'): (2.26, 4.58)},
        'EEV 8056-20-03, right': {('slow', 'low'): (2.0, 3.4), ('slow', 'high'): (4.2, 4.9), ('fast', 'high'): (5.3, 6.3), ('fast', 'low'): (2.4, 4.3)},
        'EEV 8194-19-04, right': {('slow', 'low'): (2.131, 3.93), ('slow', 'high'): (4.592, 5.46), ('fast', 'high'): (4.95, 8.41), ('fast', 'low'): (2.288, 5.25)},
        'EEV 8261-07-04, left': {('slow', 'low'): (2.056, 3.65), ('slow', 'high'): (4.381, 4.85), ('fast', 'high'): (4.868, 6.92), ('fast', 'low'): (2.264, 4.99)},
        'EEV 2037-06-03, left': {('slow', 'low'): (2.372, 3.97), ('slow', 'high'): (4.954, 5.13), ('fast', 'high'): (5.054, 6.33), ('fast', 'low'): (2.408, 4.94)},
        'EEV 2037-06-03, right': {('slow', 'low'): (2.403, 4.28), ('slow', 'high'): (4.862, 5.48), ('fast', 'high'): (5.253, 6.28), ('fast', 'low'): (2.551, 4.81)},
    }),
]


# Full-well depth of each amplifier, in electrons, for unbinned pixels (slow read-out, low gain): {DETECTOR keyword: {amplifier: depth}}
FULL_WELL = {'GMOS + e2v DD CCD42-90': {'e2v 10031-23-05, left': 110900.0,
                            'e2v 10031-23-05, right': 105100.0,
                            'e2v 10031-01-03, left': 115500.0,
                            'e2v 10031-01-03, right': 108900.0,
                            'e2v 10031-18-04, left': 115700.0,
                            'e2v 10031-18-04, right': 109200.0},
 'GMOS + Hamamatsu': {'BI5-36-4k-2, 1': 108289.0,
                      'BI5-36-4k-2, 2': 113419.0,
                      'BI5-36-4k-2, 3': 111421.0,
                      'BI5-36-4k-2, 4': 106558.0,
                      'BI11-33-4k-1, 1': 109060.0,
                      'BI11-33-4k-1, 2': 110814.0,
                      'BI11-33-4k-1, 3': 110155.0,
                      'BI11-33-4k-1, 4': 113986.0,
                      'BI12-34-4k-1, 1': 99539.4,
                      'BI12-34-4k-1, 2': 98963.6,
                      'BI12-34-4k-1, 3': 98813.4,
                      'BI12-34-4k-1, 4': 105683.0},
 'GMOS + Hamamatsu_new': {'BI5-36-4k-2, 1': 120435.0,
                          'BI5-36-4k-2, 2': 121390.0,
                          'BI5-36-4k-2, 3': 121652.0,
                          'BI5-36-4k-2, 4': 118871.0,
                          'BI11-33-4k-1, 1': 121234.0,
                          'BI11-33-4k-1, 2': 119157.0,
                          'BI11-33-4k-1, 3': 125719.0,
                          'BI11-33-4k-1, 4': 123307.0,
                          'BI12-34-4k-1, 1': 106966.0,
                          'BI12-34-4k-1, 2': 114605.0,
                          'BI12-34-4k-1, 3': 112054.0,
                          'BI12-34-4k-1, 4': 118424.0},
 'GMOS-N + Hamamatsu': {'BI13-20-4k-1, 1': 129000,
                        'BI13-20-4k-1, 2': 129000,
                        'BI13-20-4k-1, 3': 129000,
                        'BI13-20-4k-1, 4': 129000,
                        'BI12-09-4k-2, 1': 124000,
                        'BI12-09-4k-2, 2': 121000,
                        'BI12-09-4k-2, 3': 123000,
                        'BI12-09-4k-2, 4': 123000,
                        'BI13-18-4k-2, 1': 123000,
                        'BI13-18-4k-2, 2': 125000,
                        'BI13-18-4k-2, 3': 124000,
                        'BI13-18-4k-2, 4': 126000},
 'GMOS + Ham-2': {'BI11-41-4k-2, 1': 131823.0,
                  'BI11-41-4k-2, 2': 130389.0,
                  'BI11-41-4k-2, 3': 130550.0,
                  'BI11-41-4k-2, 4': 125326.0,
                  'BI13-19-4k-3, 1': 115493.0,
                  'BI13-19-4k-3, 2': 113239.0,
                  'BI13-19-4k-3, 3': 115900.0,
                  'BI13-19-4k-3, 4': 114230.0,
                  'BI12-34-4k-1, 1': 105466.0,
                  'BI12-34-4k-1, 2': 112550.0,
                  'BI12-34-4k-1, 3': 111018.0,
                  'BI12-34-4k-1, 4': 116744.0}}


# Pixel scale (arcsec per unbinned pixel): {(INSTRUME, DETTYPE): scale}
PIXEL_SCALE = {('GMOS-N', 'SDSU II CCD'): 0.0727,
 ('GMOS-N', 'SDSU II e2v DD CCD42-90'): 0.07288,
 ('GMOS-N', 'S10892-N'): 0.0807,
 ('GMOS-S', 'SDSU II CCD'): 0.073,
 ('GMOS-S', 'S10892'): 0.08}


# Relative positions of the three CCDs: {detector name (DETID keyword, without spaces after the commas... see gemini_gmos): [(x shift,
# y shift, rotation) of CCD1 and of CCD3 with respect to CCD2]}. The shifts are centre-to-centre differences, in unbinned pixels, and
# the rotations are in degrees, counterclockwise. The gap between two CCDs of 2048 pixels is therefore |x shift| - 2048.
CCD_GEOMETRY = {
    'EEV9273-16-03EEV9273-20-04EEV9273-20-03': [(-2087.5, -1.58, -0.004), (2088.8723, -1.86, -0.046)],
    'e2v 10031-23-05,10031-01-03,10031-18-04': [(-2087.7, -0.749, -0.009), (2087.8014, 2.05, -0.003)],
    'BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2': [(-2115.95, -0.21739, -0.004), (2115.48, 0.1727, -0.00537)],
    'EEV8056-20-03EEV8194-19-04EEV8261-07-04': [(-2086.44, 5.46, -0.01), (2092.53, 9.57, 0.02)],
    'EEV2037-06-03EEV8194-19-04EEV8261-07-04': [(-2086.49, -0.22, 0.011), (2089.31, 2.04, 0.012)],
    'BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1': [(-2110.2, 0.71, 0.0), (2109.0, -0.73, 0.0)],
    'BI11-41-4k-2,BI13-19-4k-3,BI12-34-4k-1': [(-2112.5, -1.1, 0.0), (2110.8, 1.05, 0.0)],
}


# Nominal gap between the CCDs (unbinned pixels), used when the detector is not in CCD_GEOMETRY
CCD_GAPS = {'EEV9273-16-03EEV9273-20-04EEV9273-20-03': 37,
 'e2v 10031-23-05,10031-01-03,10031-18-04': 37,
 'BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2': 67,
 'EEV2037-06-03EEV8194-19-04EEV8261-07-04': 37,
 'EEV8056-20-03EEV8194-19-04EEV8261-07-04': 37,
 'BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1': 61,
 'BI11-41-4k-2,BI13-19-4k-3,BI12-34-4k-1': 61}
