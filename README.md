# plaNETic
*plaNETic* is a neural network-based Bayesian internal structure modelling framework for small exoplanets with masses between 0.5 and 15 Mearth. 
It uses a full grid accept-reject sampling algorithm with neural networks trained on the interior model of the BICEPS code ([Haldemann et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract)) as a forward model. 
The code allows for different choices in priors concerning the expected abundance of water (formation inside vs. outside of iceline) and the planetary Si/Mg/Fe ratios (stellar vs. iron-enriched vs. free). 
For a more detailed description of the features of the code, we refer to Egger et al. 2024.

If you use this code, please cite Egger et al. 2024, where the *plaNETic* framework was introduced for the first time. 
If you use the trained neural networks provided, please also cite [Haldemann et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract).  

We run the code on a 2021 MacBook Pro with an Apple M1 Pro chip.  
For questions or comments, feel free to contact Jo Ann Egger (jo-ann.egger@unibe.ch).


## Example
To infer the internal structure of the planets in an observed planetary system, create a new subfolder in *run_grid* with the same structure as *TOI-469_Egger+*:
- Subfolders *posteriors*, *plots*
- Parameter file *stellar_planetary_parameters.csv* with the observed properties of the host star and all planets in the system
- Executable *run_grid_TOI-469.py*

Then simply adapt and run the executable, which will generate **_posterior.npy* files in the subfolder *posteriors*.
