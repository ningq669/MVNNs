# MVNNs

MVNNs a method integrating multi-view feature-based deep learning sub-models for S-palmitoylation sites prediction.  First, five sequence-based or structure-based protein features are extracted. Then, MVNNs build different network structures for various features.  Finally, MVNNs utilize ensemble learning and multi-view neural networks to combine all features, to generate the model for predicting S-palmitoylation sites in human proteins. 

# Requirements
  * Python 3.7 or higher
  * PyTorch 1.8.0 or higher
  * GPU (default)


# Running  the Code
  * Execute ```python new_main.py``` to run the code.

# Note
```
You can train each submodel individually, and then generate the final model through ensemble learning.
```
