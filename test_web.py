# -*- coding: utf-8 -*-

import json
import requests


def do_post(api_url, query_body):
    # headers = {'content-type': 'application/json'}
    response = requests.post(api_url, json=query_body)
    json_res = json.loads(response.text)
    print(json_res)


do_post("http://127.0.0.1:8001/embed", {"texts": ["对公付款申请，收款方银行怎么搜不到？"]})
do_post("http://127.0.0.1:8001/text_classify", {"text": "对公付款申请，收款方银行怎么搜不到？"})
