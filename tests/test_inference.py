

import pytest
from ped4you_cross_detection import predictOneTime



def test_predict(negative_img):

    # Call the predict function
    pred, prob = predictOneTime(negative_img)

    # Check if the output is as expected
    assert pred == "negative", f"Expected prediction to be 'negative', but got {pred}"
    assert 0 <= prob <= 1, f"Expected probability to be between 0 and 1, but got {prob}"

if __name__ == "__main__":
    pytest.main([__file__])