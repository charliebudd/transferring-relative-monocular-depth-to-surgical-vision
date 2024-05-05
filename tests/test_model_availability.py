import urllib.request

from trmdsv import WEIGHTS_URL


def test_model_availability() -> None:
    for model in WEIGHTS_URL:
        if not model.name.startswith("_"):
            print(f"Checking model: {model.name}, from url: {model.value}")
            response = urllib.request.urlopen(model.value)
            print(f"Response code: {response.getcode()}")
            assert response.getcode() == 200
