#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: demo.py
@time: 2023/11/15
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import json
from typing import List

import requests


def translate(intput_texts: List = None):
    if intput_texts is None:
        intput_texts = ["I love China"]
    results = []
    for text in intput_texts:
        data = {
            "prompt": "请将下面文本翻译为中文："
                      f"{text}"
        }
        post_json = json.dumps(data)
        response = requests.post("http://10.208.63.29:8888", data=post_json)  # v100-2
        answer = response.json()['response']
        results.append(answer)
    res = {
        "result": {
            "_nlu_translate": [
                results
            ]
        },
        "status": 200
    }
    return res


translate()
