The code for the plots is inside scripts/convergence_plots.py

For the dynamic selector and the one from kostovska et al., I first create the files convergence_data.csv, and convergence_data_B150.csv. This is done with create_convergence_file(). It goes through the file that contains, for each run, 
which algorithm was chosen by the selector and at which budget it switched. It then selects the corresponding run from the corresponding file in A2_run_data_test. (e.g., if for rep 3 on fid 3, instance 6, the dynamic selector switches at 250 to BFGS, we go to A2_BFGS_B250_5D.csv, and select all rows belonging to rep=3, fid=3, instance=6).

We then use aggregate_runs() to calculate, for every evaluation and fid, the mean precision of that evaluation across all runs on the fid. This gives us the entire "mean" runs per function.

This is then plotted.