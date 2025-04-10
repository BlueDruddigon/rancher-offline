import onnx

model_path = "./arcface_modify.onnx"
output_path = "./arc.onnx"

model = onnx.load(model_path)

initializer_names = [x.name for x in model.graph.initializer]

for input_tensor in model.graph.input:
    # Skip initializers (weights, biases)
    if input_tensor.name not in initializer_names:
        # This is an external input, modify its batch dimension
        if len(input_tensor.type.tensor_type.shape.dim) > 0:
            input_tensor.type.tensor_type.shape.dim[0].dim_param = "N"
            input_tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")

# Modify output
for output_tensor in model.graph.output:
    if len(output_tensor.type.tensor_type.shape.dim) > 0:
        output_tensor.type.tensor_type.shape.dim[0].dim_param = "N"
        output_tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")

# Apply shape inference on the model
inferred_model = onnx.shape_inference.infer_shapes(model)

onnx.save(inferred_model, output_path)
