import numpy as np
from tensorflow.keras.models import load_model
 
# Load model and typical SXR vector
model = load_model("/home/jussi.hakosalo/Indica/indica/workflows/TS_SXR_Interpolation.keras")
typical_sxr = np.load("/home/jussi.hakosalo/Indica/indica/workflows/mean_sxr_vector.npy")  # shape (20,)
 
def predict_ts(model, ts_start, ts_end, t, sxr_reference=None):
    """
    Predict TS value using model + optional SXR reference.
    If none provided, uses typical baseline.
 
    Args:
        model: Loaded model
        ts_start: TS at start of interval
        ts_end: TS at end of interval
        t: Normalized interpolation parameter [0, 1]
        sxr_reference: Optional SXR input (20,), or None for baseline
 
    Returns:
        float: Predicted TS value
    """
    ts_linear = ts_start * (1 - t) + ts_end * t
 
    # Use typical SXR if not provided
    sxr_input = sxr_reference if sxr_reference is not None else typical_sxr
    sxr_input = sxr_input.reshape(1, -1)
 
    t_input = np.array([[t]])
    obs_input = np.array([[ts_start, ts_end]])
 
    residual = model.predict([t_input, obs_input, sxr_input], verbose=0)[0][0]
    return ts_linear + residual
 
# Example use
if __name__ == "__main__":
    """
    The model has been trained st it learns tp operate with 2 TS observations that are 2 apart from each other. Ie.
    ts[i],ts[i+2]. The scaling parameter (last param) is [0,0.5] for the first half, [0.5,1.0] for the second half.
    """
    ts_pred1 = predict_ts(model, 1.2e19, 1.5e19, 0.5)
    ts_pred2 = predict_ts(model, 1.2e19, 1.5e19, 0.3)
    ts_pred3 = predict_ts(model, 1.2e19, 1.5e19, 0.7)
    print("Predicted TS at 0.3:", ts_pred2)
    print("Predicted TS at midpoint:", ts_pred1)
    print("Predicted TS at 0.7:", ts_pred3)