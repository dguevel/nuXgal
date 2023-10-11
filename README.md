


This is an analysis code for studying cross corellations between neturino events and galaxy samples.

Some test codes are in the 'tests' sub-directory.
The following steps install the code. 

#### Clone the repo from github
git clone https://github.com/dguevel/nuXgal.git

#### Go to the directory with the code
cd nuXgal

#### If you want the tag corresponding to the IceCube collaboration review you can checkout that version explictly. This requires access to csky (https://github.com/icecube/csky)
git checkout tags/v0.3

#### We recommed that you explictly tell the code to put IRFs and ancillary files with the code itself.  You can do this by setting:
export NUXGAL_DIR=DIRECTORY_WITH_CODE

#### Initiate a conda enviroment 
conda create -n nuXgal python=3.6 

conda activate nuXgal

#### Setup the code. For reasons I haven't figured out, the entry_points in setup.py require develop.
python setup.py develop

#### To run simulated trials use the nuXgal script which should be set up by setup.py. The other options can be left to defaults
nuXgal -i $NU_ASTRO_TO_INJECT -n $NUMBER_OF_TRIALS

#### To unblind
nuXgal --unblind

#### To plot sensitivity, fit bias, and other diagnostic plots
nuXgal-sensitivity
nuXgal-sensitivity-plot
