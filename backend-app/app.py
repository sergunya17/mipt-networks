""""
 Автор: Дмитрий Бушманов, М05-016б
 =====================================
"""

import requests
from flask import Flask, request, jsonify
from typing import List
from copy import copy

app = Flask(__name__)

CV_MODEL_URL = 'http://nn-app:5555/predict'

GOOGLE_TRANSLATOR_API_KEY = 'AIzaSyBkC-EzKR-XtJzJgihWnXKfmhMCPDQgIks'
GOOGLE_TRANSLATOR_URL = 'https://translation.googleapis.com/language/translate/v2'
DETECT_URL = '/image/recognize/detect-objects'

FOREIGN_LANG_JSON_FIELD_NAME = 'foreign_lang'
NATIVE_LANG_JSON_FIELD_NAME = 'native_lang'
IMAGE_JSON_FIELD_NAME = 'imageFile'

""""
 Запрос к Google Translation API

 text: Array of String for translation
 target_language: target translation language
"""


def translate_request(text: List[str], target_language: str) -> List[str]:
    response = requests.post(GOOGLE_TRANSLATOR_URL,
                             json={'q': text, 'target': target_language},
                             params={'key': GOOGLE_TRANSLATOR_API_KEY})
    json_response = response.json()
    translate_list = []
    for element in json_response.get('data').get('translations'):
        translate_list.append(element.get('translatedText'))
    return translate_list


# Add additional fields to Json
def add_translation_to_json(json_obj: list, translate_list: List[str], lang_type: str) -> list:
    i = 0
    for elem in translate_list:
        json_obj['Objects'][i][lang_type] = elem
        i += 1
    return json_obj


"""
 Main controller method
"""


@app.route(DETECT_URL, methods=['POST'])
def detect_objects():
    # Параметры запроса
    if not request.args.get(FOREIGN_LANG_JSON_FIELD_NAME):
        return FOREIGN_LANG_JSON_FIELD_NAME + ' not found in request parameters.', 400
    else:
        foreign_lang = request.args[FOREIGN_LANG_JSON_FIELD_NAME]

    if not request.args.get(NATIVE_LANG_JSON_FIELD_NAME):
        return NATIVE_LANG_JSON_FIELD_NAME + ' not found in request parameters.', 400
    else:
        native_lang = request.args[NATIVE_LANG_JSON_FIELD_NAME]

    # Загруженное изображение
    if not request.files.get(IMAGE_JSON_FIELD_NAME):
        return IMAGE_JSON_FIELD_NAME + ' not found in request body.', 400
    else:
        file = request.files[IMAGE_JSON_FIELD_NAME]

    # Запрос к нейросети
    response = requests.post(CV_MODEL_URL,
                             files={IMAGE_JSON_FIELD_NAME: ("imageFile.jpg", file, 'multipart/form-data', {'Expires': '0'})})
    json_response = response.json()

    if bool(json_response.get('Successful')):
        # Запрос к Google Translation API
        objects_name = []
        for elem in json_response['Objects']:
            objects_name.append(elem['ObjectClassName'])

        foreign_lang_translation_list = translate_request(objects_name, foreign_lang)
        native_lang_translation_list = translate_request(objects_name, native_lang)

        json_new_response = copy(json_response)
        json_new_response = add_translation_to_json(json_new_response, foreign_lang_translation_list, FOREIGN_LANG_JSON_FIELD_NAME)
        json_new_response = add_translation_to_json(json_new_response, native_lang_translation_list, NATIVE_LANG_JSON_FIELD_NAME)

        return json_new_response, 200
    else:
        return jsonify({'Successful': 'False'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
