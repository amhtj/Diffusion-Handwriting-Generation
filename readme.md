Implementation of the code from the paper "Diffusion Models for Handwriting Generation": https://arxiv.org/abs/2011.06704

A number of errors have been found, most of which are related to how the Tensorflow has changed (2.3 vs. 2.12). Internal mathematical functions have been changed, due to which we observe a discrepancy in the final sizes of tensors. 

The model was launched on a HSE supercomputer, via a connection to Mountain Duck. Unfortunately, the problem with the slash did not allow to continue training, which is why the calculations were transferred to another supercomputer (VTB). 

Requirements from the authors: python 3.7, Tensorflow 2.3
My requirements: Python 3.11, TF 2.12.

I retrained the model as a research, as I could not teach her into Russian.This is due to the fact that it took a long time to fix errors in the code. Weights are uploaded via the link to my GDrive (too large for repo):

https://drive.google.com/file/d/1Cvkd1Iq-r-rvH9HO8vgCap4Lzt7aAi-r/view?usp=share_link

To get an inference, run inference.py (with arguments, see the file). To retrain model run train.py. Arguments can be changed. 

IAM Dataset from this project can be downloaded only with VPN (IAM dataset represented by a Swiss laboratory).

https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

We need the following files:

data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset

After that, we extracting these contents and put them in the ./data directory. 

The authors calculated the FID metric, I rewrote it (library incompatibility prompted this) and added the Inseption, CLIP calculation.
