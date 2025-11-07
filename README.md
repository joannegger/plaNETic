![plaNETic Logo](logo/plaNETicLogo_reduced.png)

*plaNETic* is a neural network-based Bayesian internal structure modelling framework for small exoplanets with masses between 0.5 and 15 Mearth. 
The code efficiently computes posteriors of a planet's internal structure based on its observed planetary and stellar parameters. 
It uses a full grid accept-reject sampling algorithm with neural networks trained on the interior model of the [BICEPS code](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract) as a forward model. 
Furthermore, it allows for different choices in priors concerning the expected abundance of water (formation inside vs. outside of iceline) and the planetary Si/Mg/Fe ratios (stellar vs. iron-enriched vs. free).  

For a more detailed description of the features of the code, we refer to [Egger et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240618653E/abstract), where the code was first introduced and applied to a planetary system.

We run the code on a 2021 MacBook Pro with an Apple M1 Pro chip.  
For questions or comments, feel free to contact Jo Ann Egger (joann.egger@esa.int).

## Citations
If you use this code, please cite [Egger et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240618653E/abstract), where this version of the *plaNETic* framework was introduced for the first time.  
If you use the trained neural networks provided, please also cite [Haldemann et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...681A..96H/abstract).  

An up-to-date list of publications that are using the plaNETic code can be found in [this ADS library](https://ui.adsabs.harvard.edu/public-libraries/1gUfHn6dR5qTwZ9phcfNMg).  
If you used the code for your own publication, reach out to us and we will add it to the list!  

## Installation
git clone plaNETic and run "pip install ."

If downloading the DNNs causes an error, please contact Jo Ann Egger (joann.egger@esa.int) while we are looking for a more permanent data storage solution.

If you are running the code on a MacBook with an Apple silicon chip, make sure tensorflow has access to the built-in GPU for a much faster code execution. 
You can follow for example this tutorial: https://medium.com/bluetuple-ai/how-to-enable-gpu-support-for-tensorflow-or-pytorch-on-macos-4aaaad057e74

## Example
To infer the internal structure of the planets in an observed planetary system, create a new subfolder in *run_grid* with the same structure as *TOI-469_Egger+*:
- Subfolders *posteriors*, *plots*
- Parameter file *stellar_planetary_parameters.csv* with the observed properties of the host star and all planets in the system
- Executable *run_grid_TOI-469.py*

Then simply adapt and run the executable, which will generate **_posterior.npy* files in the subfolder *posteriors*.
