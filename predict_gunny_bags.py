"""
Script to predict the number of gunny bags in an input image using a pre‑trained
regression model.  The model was trained on a small set of jute roll and bag
images and therefore should be considered a demonstration rather than a
production quality solution.  Given an image file path on the command line,
the script resizes the image to 32×32 pixels, flattens it, scales pixel
intensities to the 0‑1 range and feeds it into a scikit‑learn regression
model stored in ``gunny_bag_count_model.pkl``.  The predicted count is
printed to stdout as an integer.

Usage:
    python predict_gunny_bags.py path/to/image.jpg

If multiple image paths are provided, predictions will be printed for each
in the order they appear.
"""

import sys
import os
import pickle
from PIL import Image
import numpy as np

# Default model file.  This version uses a simple Ridge regression model
# saved with joblib.  Ridge models tend to serialize in a version‑stable
# way across scikit‑learn releases.  You can replace this filename with
# the path to a model you trained yourself using ``train_gunny_bags.py``.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gunny_bag_count_model_ridge.pkl')


def load_model(path: str):
    """Load a scikit‑learn regression model from disk.

    The model is stored in joblib format which offers better cross‑version
    compatibility than pickle.  If joblib is unavailable for some reason,
    it falls back to pickle.
    """
    # Try joblib first
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception:
        # Fall back to pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


def preprocess_image(img_path: str) -> np.ndarray:
    """Load an image, resize to 32×32 and flatten to a 1D vector.

    Parameters
    ----------
    img_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Flattened image vector with values in the range [0.0, 1.0].
    """
    # Open image in RGB mode. PIL handles most formats including JPEG and WEBP.
    img = Image.open(img_path).convert('RGB')
    # Resize image to match the training resolution
    img = img.resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def predict_count(model, image_vector: np.ndarray) -> int:
    """Predict the number of gunny bags given a flattened image vector.

    The model predicts a continuous value which is then rounded to the nearest
    non‑negative integer.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A regression model supporting the ``predict`` method.
    image_vector : numpy.ndarray
        Flattened representation of the image.

    Returns
    -------
    int
        Predicted number of gunny bags, clamped to a minimum of zero.
    """
    # scikit‑learn models expect 2D arrays for prediction
    prediction = model.predict([image_vector])[0]
    # Round the prediction and clamp to non‑negative values
    return max(int(round(prediction)), 0)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: python predict_gunny_bags.py <image1> [<image2> ...]")
        return 1
    # Ensure model file exists
    if not os.path.exists(MODEL_PATH):
        sys.stderr.write(f"Model file not found: {MODEL_PATH}\n")
        return 1
    # Load model once
    model = load_model(MODEL_PATH)
    for img_path in argv:
        try:
            vec = preprocess_image(img_path)
            count = predict_count(model, vec)
            print(f"{os.path.basename(img_path)}: {count}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())