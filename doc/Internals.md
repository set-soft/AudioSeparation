# Internals

## Call sequence

- The node creates a demixer class with: get_demixer(model_data, device, models_dir)
- This function creates a Demixer object for the correct demix type
- The DemixerGeneric.__init__() calls load_model
- load_model:
  - Calls get_model to get a proper model object without weights
  - Calls a loader for the container (ONNX, safetensors)
    - load_safetensors loads the weights
