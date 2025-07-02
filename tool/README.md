# Audio separatio tools

Some random tools used during development.

They are intended to be invoked from the root of the nodes.

# ONNX to safetensors converter

File: onnx2safetensors.py

Used to convert MDX-Net models in ONNX format to safetensors.
Might work for other ONNX files using the same operands.
But you'll need a PyTorch class for its architecture that creates the layers
in the same order as the ONNX file.

# Batch converter

File: batch_convert.py

Used to convert all the MDX-Net files in one run

# Demix

File: demix.py

Used to run the inference on audio files performing the demixing process

# Download Model

File: download_model.py

Downloads a model listed in the DB

# Show Data Base

File: show_db.py

Prints the content of the models data base.
This is useful to adjust details in the DB, but you need to modify the code (`apply_process`)

# Show ONNX

File: show_onnx.py

Displays the ONNX input, output and layers.
Is a text representation, not as nice as [Netron](https://netron.app/), but you
can extract a lot of information from it.

It can also show details of what onnx2pytorch interprets from it.

And can do an inferece run using random data. We used it to determine some
details of the onnx2pytorch details.

# Show Class

File: show_class.py

Used to display our PyTorch class, also the state_dict keys.
It can optionally export the class structure as an ONNX file that can be loaded by
Netron, but contains too much extra names.

# Style Fixer

File: style_fixer.py

Fixes various common style errors found in Gemini 2.5 Pro code.
The script was created by Gemini ... can create the script, but can't stop making
the same errors over and over.

# UVR Hash

File: uvr_hash.py

Computes the hash used by UVR to identify a model. We use the same hash, as many
tools do.

This is the key value for aur model database.
