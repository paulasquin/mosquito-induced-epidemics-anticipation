from image_recognition.inception_classification.utilities.EnvReader import get_api_key
from image_recognition.inception_classification.utilities.Errors import EnvError

if not isinstance(get_api_key(), str):
    raise EnvError()
else:
    print("Success!")
