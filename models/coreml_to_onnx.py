"""
Converting CoreMl format to ONNX format.
Run just by Python=2.7. CoreML not supported in newest versions of Onnxmltools.
"""
import coremltools
import onnxmltools


def core_onnx(core_model: str, path_to_save: str) -> None:
    """
    Converts CoreML model into ONNX format

    Parameters
    ----------
    core_model : str
        Path to CoreML model
    path_to_save : str
        Path to save ONNX model
    """
    coreml_model = coremltools.utils.load_spec(core_model)

    onnx_model = onnxmltools.convert_coreml(coreml_model, "Linear regression model")
    onnxmltools.utils.save_model(onnx_model, path_to_save)


if __name__ == "__main__":
    core_onnx(
        "/root/projects/ml-selection/models/mathematical_models/linear_regression_model.mlmodel",
        "/models/onnx/linear_regression_model.onnx",
    )
