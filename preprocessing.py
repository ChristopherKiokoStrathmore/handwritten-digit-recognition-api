"""
preprocessing.py  —  the SINGLE source of truth for turning a raw image
into what the model expects.

Why this file exists (the most important lesson in production ML):
Your model was TRAINED on images scaled to 0-1 and flattened to 784 numbers.
If your SERVER scales them even slightly differently, the model quietly gets
worse and NOTHING crashes to warn you. This is called "training/serving skew".
The defence is simple: training and serving both import THIS function. There
is only one copy of the logic, so they can never drift apart.
"""

import numpy as np

# The model's input contract, stated once, in one place.
IMAGE_SIZE = 28          # MNIST images are 28x28
FLAT_SIZE = 28 * 28      # 784 — the flattened length the dense model wants


def preprocess_image(raw_grid):
    """
    Turn a raw 28x28 grid of pixel values (0-255) into a model-ready
    (1, 784) float32 array scaled to 0-1.

    raw_grid: a 28x28 nested list or numpy array, values 0-255.
    returns:  numpy array of shape (1, 784), values 0.0-1.0.
    raises:   ValueError if the shape is wrong (this is a FEATURE — we want
              a loud, clear error, not a silent bad prediction).
    """
    arr = np.asarray(raw_grid, dtype="float32")

    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, but got shape {arr.shape}. "
            f"Refusing to guess."
        )

    # THE scaling. Identical to Task 1.4 of the notebook. Divide by 255.
    arr = arr / 255.0

    # Flatten the 28x28 grid into one row of 784, and add a batch dimension
    # so the shape is (1, 784) — the model always expects a batch.
    return arr.reshape(1, FLAT_SIZE)


# A tiny self-test so you can run `python preprocessing.py` and confirm it works.
if __name__ == "__main__":
    fake = np.random.randint(0, 256, size=(28, 28))
    out = preprocess_image(fake)
    print("Input shape :", fake.shape)
    print("Output shape:", out.shape, "(should be (1, 784))")
    print("Value range :", round(float(out.min()), 3), "to", round(float(out.max()), 3),
          "(should be within 0.0 to 1.0)")
    try:
        preprocess_image(np.zeros((32, 32)))
    except ValueError as e:
        print("Correctly rejected a wrong-sized image:", str(e)[:60], "...")
