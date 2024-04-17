"""
Converting coreml format to onnx format.
Run just by Python=2.7. CoreML not supported in newest versions of Onnxmltools.
"""
import coremltools
import onnxmltools


def core_onnx(core_model, path_to_save):
    """
    Converts CoreML model into ONNX format

    Parameters
    ----------
    core_model : str
        Path to CoreML model
    path_to_save : str
        Path to save ONNX model
    """
    c_model = str(core_model)
    coreml_model = coremltools.utils.load_spec(c_model)

    onnx_model = onnxmltools.convert_coreml(coreml_model, "Linear regression model")
    onnxmltools.utils.save_model(onnx_model, path_to_save)
    return onnx_model


if __name__ == "__main__":
    model = core_onnx(
        "/root/projects/ml-selection/models/mathematical_models/linear_regression_model.mlmodel",
        "/root/projects/ml-selection/models/mathematical_models/linear_regression_model.onnx",
    )
