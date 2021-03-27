# DQN Experiments with multi-step learning
command: python3 run_experiment.py --nstep *n*

Experiments will be saved in experiment_results.
Models can be saved via run.py and executed via run_model.py

/lib offers some implementations from myself like *DQN* or *replay memory* and some 3rd party implementations 
like *wrappers* from openai or *MultiStepBuffer* from pytorch lightning.

The scripts in /visualizations build figures on top of the results from the experiments.
