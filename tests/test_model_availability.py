import urllib.request

from trmdsv import MODEL


def test_model_availability() -> None:
    for model in MODEL:
        if not model.name.startswith("_"):
            print(f"Checking model: {model.name}, from url: {model.value}")
            response = urllib.request.urlopen(model.value)
            print(f"Response code: {response.getcode()}")
            assert response.getcode() == 200
