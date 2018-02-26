# SimATRP

This is a controllable Atom Transfer Radical Polymerization (ATRP) simulator, based on solving the ATRP kinetics ordinary differential equations (ODEs).  By "controllable" it means that, beyond the level of control in the sense of "controlled polymerization", it provides another, "mechanical-ish", level of control where the addition of ATRP reagents is partitioned into many small minibatches, and a well-optimized controller may operate certain control sequences onto the simulator based on monitoring the simulator status in realtime in order to fine-tune the final product molecular weight distributions (MWDs) into (arbitrarily specified) shapes.

To get a sense of how the control process would look like, please run `simatrp_interactive.py` as a script.  This script is found in your `$PATH` after installing this package, or under the `bin` folder in the source code package.

## Requirements
```
numpy
scipy
gym>=0.9.6
matplotlib
pygame
h5py
```
