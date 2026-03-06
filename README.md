![plaNETic Logo](logo/plaNETicLogo_reduced.png)

plaNETic is a neural network-based Bayesian internal structure modelling framework for small exoplanets with masses between 0.5 and 15 Mearth. 
The code efficiently computes posteriors of a planet's internal structure based on its observed planetary and stellar parameters. 
It uses a full grid accept-reject sampling algorithm with neural networks trained on the interior model of the [BICEPS code](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract) as a fast surrogate for the forward model. 
Furthermore, it allows for different choices in priors concerning the expected abundance of water (formation inside vs. outside of iceline) and the planetary Si/Mg/Fe ratios (stellar vs. iron-enriched vs. free).  

For a more detailed description of the features of the code, we refer to [Egger et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240618653E/abstract), where the code was first introduced and applied to a planetary system. 
An up-to-date list of publications that are using the plaNETic code can be found in [this ADS library](https://ui.adsabs.harvard.edu/public-libraries/1gUfHn6dR5qTwZ9phcfNMg).  

## Using this code
If you want to use this code, please get in contact with Jo Ann Egger (joann.egger@esa.int). The plaNETic team is happy to collaborate on your project, especially by providing support with running the code and helping to interpret the results.  

If you use the plaNETic framework, please cite [Egger et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240618653E/abstract), where this version of the code was introduced for the first time. If you also use the trained neural networks provided, please also cite [Haldemann et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract).  

## Installation
plaNETic needs access to a GPU to run efficiently. While it is in principle possible to run the code on CPUs only, this will drastically increase the computation time and is not recommended.  

### On a MacBook with an Apple Silicon chip
To install the code, create a new virtual environment with Python 3.9, then git clone plaNETic and run 'pip install .'  
You can now run one of the example scripts and check that everything is running correctly. For this check, we recommend running plaNETic for TOI-238 b, see description below.  

To make sure tensorflow has access to the built-in GPU, you can follow for example this tutorial: https://medium.com/bluetuple-ai/how-to-enable-gpu-support-for-tensorflow-or-pytorch-on-macos-4aaaad057e74  

### On Windows
Installing plaNETic on Windows is a bit more tedious but possible. Reach out to us if you need support with this, we are happy to share a protocol of steps that worked for us.

## Running plaNETic
To infer the internal structure of the planets in an observed planetary system, create a new subfolder in 'run_grid' with the same structure as 'TOI-469_Egger+' or 'TOI-238_Egger+':
- Parameter file 'stellar_planetary_parameters.csv' with the observed properties of the host star and all planets in the system that you would like to model
- Executable 'run_grid_xxx.py'

Then simply adapt and run the executable, which will generate 'xxx_posterior.npy' files in a subfolder 'posteriors'.  

If plaNETic cannot find the trained DNNs, this is an issue with GitHub's Large File Storage. In this case, please contact Jo Ann Egger (joann.egger@esa.int) while we are looking for a more permanent data storage solution.  

Depending on your machine, changing the batch size when running the neural network (in the function 'compute_radius', line 1308 of 'plaNETic.py') might improve the performance of the code.  

## Example
To ensure the code is running correctly, we recommend running it for TOI-238 b. To do that, navigate into 'run_grid/TOI-238_Egger+' and run the executable 'run_grid_TOI-238.py' with Python. On a MacBook with an M4 Pro Apple Silicon chip, this gives us the following output:  

```
Metal device set to: Apple M4 Pro  
  
systemMemory: 48.00 GB  
maxCacheSize: 18.00 GB  
  
Option A1:  
100%|████████████████████████| 9429/9429 [13:41<00:00, 11.47it/s]  
Shape of posteriors:  
(1, 771909, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1979.29it/s]  
Shape of priors:  
(1, 117814, 28)  
  
Option A2:  
100%|████████████████████████| 9429/9429 [14:37<00:00, 10.74it/s]  
Shape of posteriors:  
(1, 1941179, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1890.20it/s]  
Shape of priors:  
(1, 140433, 28)  
  
Option A3:  
100%|████████████████████████| 9429/9429 [11:17<00:00, 13.92it/s]  
Shape of posteriors:  
(1, 855115, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1776.27it/s]  
Shape of priors:  
(1, 72619, 28)  
  
Option B1:  
100%|████████████████████████| 9429/9429 [15:24<00:00, 10.20it/s]  
Shape of posteriors:  
(1, 16505, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1728.45it/s]  
Shape of priors:  
(1, 156788, 28)  
  
Option B2:  
100%|████████████████████████| 9429/9429 [16:54<00:00,  9.29it/s]  
Shape of posteriors:  
(1, 1113545, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1643.45it/s]  
Shape of priors:  
(1, 187370, 28)  
  
Option B3:  
100%|████████████████████████| 9429/9429 [12:18<00:00, 12.77it/s]  
Shape of posteriors:  
(1, 570407, 28)  
100%|████████████████████████| 939/939 [00:00<00:00, 1607.09it/s]  
Shape of priors:  
(1, 96893, 28)
```

As shown above, the inference process for one planet and prior option takes around 10-15 minutes on our machine. In multiplanetary systems, all planets are modelled simultaneously, with the computation time scaling linearly with the number of planets.
