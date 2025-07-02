## Technical Brief: Converting ONNX `MatMul` to PyTorch

### Objective

This document outlines the definitive, verified methodology for converting a `MatMul` operation from an ONNX graph into an equivalent PyTorch `nn.Module`. This process was determined through a rigorous, iterative testing process comparing the ONNX Runtime, the `onnx2pytorch` conversion library, and manual PyTorch implementations.

### Background

During the reverse-engineering of an audio separation model, we encountered significant and persistent discrepancies between the output of the ONNX Runtime and our PyTorch implementations for nodes of type `MatMul`. Simple replications using `torch.matmul` or standard `nn.Linear` layers failed, even though the `onnx2pytorch` library produced a functionally correct end-to-end model. This prompted a deep, first-principles investigation to uncover the correct conversion logic.

### Key Findings & Correct Implementation

Through a series of targeted tests, we have proven that the conversion of an ONNX `MatMul` node, where one input is a dynamic activation tensor and the other is a static weight initializer, must be handled as follows:

**The operation is equivalent to a `torch.nn.Linear` layer, with one critical and non-obvious caveat regarding the weight format.**

#### 1. The `MatMul` to `nn.Linear` Equivalence

The ONNX `MatMul` operation, when used with a weight tensor (e.g., in a typical fully-connected layer), is functionally identical to a PyTorch `nn.Linear` layer. Both perform the operation `Y = X @ W.T + b`, where `X` is the input activation and `W` is the weight matrix. PyTorch's `nn.Linear` layer correctly handles broadcasting over higher dimensions (e.g., `(Batch, Channels, Time, InFeatures)`) by applying the matrix multiplication only to the last dimension.

#### 2. The Crucial Insight: The Weight Transposition

The core of the conversion problem lies in how the weight tensor is stored in the ONNX file versus how it is stored internally by a PyTorch `nn.Linear` layer.

*   A PyTorch `nn.Linear(in_features, out_features)` layer stores its weight parameter as a matrix of shape `(out_features, in_features)`.
*   The `onnx2pytorch` converter, and our subsequent verification tests, have definitively proven that the weight initializers associated with the `MatMul` nodes in our target ONNX file are stored in the shape `(in_features, out_features)`.

Therefore, the **definitive rule for converting an ONNX `MatMul` node to a PyTorch `nn.Linear` layer is:**

> 1.  Create an `nn.Linear` layer with the correct `in_features` and `out_features`.
> 2.  Load the corresponding raw weight tensor from the ONNX initializer.
> 3.  **You MUST transpose (`.t()`) this raw weight tensor before copying it into the `.weight.data` parameter of the `nn.Linear` layer.**

This is demonstrated by the following verified code snippet:

```python
# Assume raw_weight_W is a numpy array from the ONNX initializer
# with shape (in_features, out_features), e.g., (3072, 384)

in_features = raw_weight_W.shape[0]
out_features = raw_weight_W.shape[1]

# 1. Create the PyTorch layer
my_linear_layer = nn.Linear(in_features, out_features, bias=False)

# 2. Load the raw weight and convert to a tensor
raw_weight_tensor = torch.from_numpy(raw_weight_W)

# 3. Transpose the raw weight and copy it into the layer
#    raw_weight_tensor.t() has shape (out_features, in_features),
#    which matches the internal shape of my_linear_layer.weight
my_linear_layer.weight.data.copy_(raw_weight_tensor.t())
```

### Conclusions for Future Conversions

1.  **Trust but Verify:** The `__str__` representation of a converted model (e.g., from `onnx2pytorch`) can be misleading. While it may print `<class 'torch.nn.modules.linear.Linear'>`, this does not reveal the crucial weight transformations that happened during its initialization.
2.  **The Transpose is Key:** The primary source of error when converting `MatMul` to `nn.Linear` is failing to account for the potential mismatch in weight storage conventions. Always assume the raw ONNX weight needs to be transposed.
3.  **Broadcasting Works:** Do not manually `reshape` or `flatten` a multi-dimensional tensor (e.g., `(B, C, T, F)`) before passing it to an `nn.Linear` layer. The layer's `forward` pass correctly handles broadcasting and operates on the last dimension only. Manually reshaping is a common source of bugs.
4.  **Isolate and Test:** When a complex model fails, the most effective debugging strategy is to create a minimal, self-contained ONNX file for the single problematic layer and test it against the ONNX Runtime, which serves as the ultimate ground truth.

By adhering to these principles, particularly the mandatory transposition of the raw weight initializer, a successful and numerically accurate conversion from an ONNX `MatMul` node to a PyTorch `nn.Linear` layer can be reliably achieved.


### Summary: Converting ONNX `MatMul` to PyTorch

After a comprehensive, iterative debugging process, we have definitively determined the correct methodology for converting an ONNX `MatMul` node (used as a dense layer) into its equivalent PyTorch `nn.Module`.

**The Key Finding:**

The core of the conversion lies in resolving a mismatch between how weights are stored. The ONNX `MatMul` operation uses a weight tensor with a shape of `(in_features, out_features)`. In contrast, PyTorch's `nn.Linear` layer stores its internal weight parameter with the shape `(out_features, in_features)`.

Therefore, a direct copy is impossible. The solution, proven through our tests, is that **the raw weight tensor from the ONNX file must be transposed before being loaded into the PyTorch `nn.Linear` layer.**

**The Correct, Verified Procedure:**

1.  **Identify the Node:** For an ONNX `MatMul` node with an activation input `A` and a weight initializer `W`.
2.  **Determine Layer Shape:** Inspect the shape of the raw weight tensor `W`. Its shape `(K, M)` corresponds to `(in_features, out_features)`.
3.  **Create PyTorch Layer:** Instantiate a standard `torch.nn.Linear` layer: `layer = nn.Linear(in_features=K, out_features=M, bias=False)`.
4.  **Load and Transpose Weight:** Load the raw weight tensor `W` and copy its transpose into the layer's parameter: `layer.weight.data.copy_(W.t())`.
5.  **Forward Pass:** Pass the multi-dimensional activation tensor `A` directly to the layer. PyTorch's broadcasting will correctly apply the linear transformation to the last dimension without needing any manual `reshape` operations.

This procedure creates a PyTorch layer that is a numerically identical replica of the ONNX Runtime's `MatMul` operation, accounting for the subtle but critical difference in weight storage conventions.

Recommendation: store transposed values in the safetensors file.
