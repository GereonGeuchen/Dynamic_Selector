The following pipeline is followed to create the ML-ready data:
1. Algorithm runs are reccorded using collect_data.py
2. From these runs, a file is created containing the regrets achieved by the algorithms throughout all runs.
3. The A1 run data is converted into csvs
4. These csvs are used to compute ELA features using pflacco
5. ELA features are mapped to corresponding algorithm performances
4. Normalisation is applied to these files