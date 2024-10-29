"""
Run with command:
$ python PATH_TO_SCRIPT PATH_TO_PROJECT PATH_TO_MODEL_IN_ONNX_FORMAT PATH_TO_TXT_FILE OUTPUT_PATH

File for input must consist of: "poly_elements", "poly_type", "temperature".
All data is separate by comma.

For example:

Cu<sub>4</sub>Al<sub>11</sub>,15-a,298
Cu<sub>4</sub>Al<sub>128</sub>,11-a,298
"""

import os
import sys

import numpy as np
import onnxruntime as rt

path_to_project = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[4]
sys.path.append(os.path.abspath(path_to_project))


from data_massage.create_polyheadra import (
    get_int_poly_type,
    get_poly_elements,
    size_customization,
)


def run_predict(input: np.array):
    """Input size is (num_examples, 103)"""
    sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    pred = sess.run(None, {"input": input.astype(np.float32)})
    return pred[0]


# txt file with values: "poly_elements", "poly_type", "temperature",
# for example:
# Cu<sub>4</sub>Al<sub>11</sub>,15-a,298
with open(sys.argv[3], "r") as file:
    structures = [line.replace(r"\n", "").split(",") for line in file]

inputs = []

for structure in structures:
    elements = get_poly_elements(structure, idx=0)
    vertex, p_type = get_int_poly_type(structure, idx=1)
    elements_large = size_customization(elements)
    temperature = int(structure[2])
    inputs.append(elements_large + [vertex] + [p_type] + [temperature])

output = run_predict(np.array(inputs))

with open(output_path, "w") as output_file:
    for i, out in enumerate(output):
        output_file.write(str(output[i][0]) + "\n")

[print(i) for i in structures]
print("=" * 100)
[print(out[0]) for out in output]
