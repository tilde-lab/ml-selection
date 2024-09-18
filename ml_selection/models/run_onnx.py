import onnxruntime as rt


def run_model_onnx(model_onnx: str, atom: list, distance: list) -> list:
    """
    Run Linear Regression model in ONNX format

    Parameters
    ----------
    model_onnx : str
        Path to model in ONNX format
    atom : list
        Samples of feature 'atom'.
        One sample consist of list with 100 periodic number of atom, which is wrapped in a string.
    distance : list
        Samples of feature 'distance'

    Returns
    -------
        Prediction of Seebeck value
    """
    preds = []

    sess = rt.InferenceSession(model_onnx, providers=rt.get_available_providers())

    for i in range(len(atom)):
        pred_onx = sess.run(None, {"atom": [atom[i]], "distance": [distance[i]]})
        preds.append(pred_onx[0][0][0])

    return preds
