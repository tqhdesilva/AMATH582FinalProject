# AMATH582FinalProject

### Project Ideas
- Ibis Network Data(contingent on getting their permission): 
    IoT data from smart sockets.
    Their take home technical involved detecting when a device is being used.
    Power consumption background level can be noisy and variable, and usage is indicated
    by quick increase in power level consumption before returning to normal background level.
    I used some GMM and time series methods to complete their take home, but I'd like to try
    to apply some wavelet and time signal techniques to the problem.
- VSB Powerline Fault Detection: Detect partial discharge patterns in signals acquired from faulty power lines.
    This is a classification problem for a kaggle competition a year ago. Each measurement has 800,00 samples over 20
    milliseconds. Three phases are measured simultaneously. There are over 8000 signals in the train set. I want to
    see how I can apply Fourier transform and wavelets to this problem.
- NLOS f-k migration data set related to [this](http://www.computationalimaging.org/publications/nlos-fk/)
    research project. Not sure if we have time to really implement some of the stuff in this paper or even understand it,
    but looks cool. Might be nice to revisit this since learning about filtering and wavelets.
- Talk with Jason about DICOM data he was looking at before.
- LANL Earthquake prediction: Another kaggle competition data set. Apply wavelets and signal analysis to this.
- [DICOM](http://headctstudy.qure.ai/#dataset)
