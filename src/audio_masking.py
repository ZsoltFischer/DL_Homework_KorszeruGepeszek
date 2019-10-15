'''
This module implements basic functions for working with the STFT-form
of audio signals, for use in multi-talker separation.

The IRM-s are used for generating the training sets, and the reconstruction
of the spectrum is used for applying the masks on the mixed speech to generate
separate talker signals. After masking, the resulting data can be converted
back to audio signals using inverse STFT.

Common notation:
X1, X2, ... : STFT-form of separate audio signals with separate speakers
Y: sum(X1, X2, ...), the STFT of the mixed signal (STFT is a linear operator)
M1, M2, ... : masks for speakers to be able to separate talkers from Y

All inputs and outputs are if type np.ndarray (dtype=complex64 by default).
'''

import numpy as np


def reconstructSpectrum(Ms, Y):
  # Based on the mixed signal Y, this function separates out a speaker
  # using a mask Ms. An important caveat here that Ms only separates the 
  # amplitude values of Y, the phase values of Y are used without masking.

  # input: Ms: a (not neccessarily ideal) mask that corresponds to a speaker
  #        Y: the mixed speech
  # output: Xs: the separated STFT values corresponding to one speaker
  XsAbs = Ms*np.abs(Y)
  return XsAbs*(np.e)**(1j*np.angle(Y))

def calculateY(Xs):
  # input: Xs = [X1, X2, ...] separate STFT values for speakers (complex valued)
  # output: Y = the STFT of the mixed audio
  Y = np.zeros(Xs[0].shape, dtype=Xs[0].dtype)
  for i in range(len(Xs)):
    Y+=Xs[i]
  return Y

def IRM(Xs, Y):
  # Calculates the Ideal Ratio Masks for all speakers per Yu et al 2016.

  # input: Xs = [X1, X2, ...]: separate STFT values for speakers (complex valued)
  #        Y: the STFT values of the mixed speech
  # output: Ms = [M1, M2, ...]: separate IRM masks for all input speakers

  newShape = len(Xs), *Xs[0].shape
  Ms = np.zeros(newShape, dtype=Xs[0].dtype)

  for i in range(len(Xs)):
    Ms[i] = np.abs(Xs[i])/np.abs(Y)
  return Ms

def IRMbeta(Xs, beta=0.5):
  # Calculates the Ideal Ratio Masks for all speakers per Wang 2014.
  # This seems to be a slightly more accurate approach.

  # input: Xs = [X1, X2, ...]: separate STFT values for speakers (complex valued)
  #        beta: a parameter for generating the IRM; Wang 2014 recommend beta=0.5.
  # output: Ms = [M1, M2, ...]: separate IRM masks for all input speakers
  newShape = len(Xs), *Xs[0].shape
  Ms = np.zeros(newShape, dtype=Xs[0].dtype)
  
  XsSquareSum = np.zeros(Xs[0].shape, dtype=Xs[0].dtype)
  
  for Xi in Xs:
    XsSquareSum+=Xi**2
  for i in range(len(Xs)):
    Ms[i] = Xs[i]**2/XsSquareSum
  
  return Ms**beta