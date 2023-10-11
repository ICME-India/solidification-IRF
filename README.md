# solidification-IRF
This is a repository for python scripts to calculate Interface Response Functions for solidification. Interface response functions with and without Calphad coupling are calculated. More details can be found in our preprint [http://arxiv.org/abs/2303.07663](http://arxiv.org/abs/2303.07663) titled "Interface Response functions for multicomponent alloys - An Application to additive manufacturing". The codes will updated once the manuscript is accepted for publication.

## Running the script
All the models have the format for running. All the parameters are mentioned in a `json` file and used as a argument for each model. The values needed in each model is different and hence it is advisable to use separate files for each model. When run in a terminal, it can be run as follows
`python3 run_KGT.py H13_delta.json`
For running in Jupyter Notebook,
`%run run_KGT.py H13_delta.json`
For running in spyder,
`runfile("run_KGT.py",args="H13_delta.json")`  
For Calphad-coupled models (Ludwig and Wang), `tcpython` API of Thermo-Calc needed. Although this dependency will be removed in future and options for using `pycalphad` will be enabled.
