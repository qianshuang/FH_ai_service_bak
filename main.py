# -*- coding: utf-8 -*-

import os
from emb_helper import *
from finetune_llm_helper import *

from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/embed")
async def embed(request: Request):
    print("{} is processing...".format(os.getpid()))

    json_post_raw = await request.json()
    texts = json_post_raw["texts"]

    embs = get_batch_emb(texts)
    return {"code": 0, "msg": "success", "data": embs}


@app.post("/text_classify")
async def text_classify(request: Request):
    print("{} is processing...".format(os.getpid()))

    json_post_raw = await request.json()
    text = json_post_raw["text"]

    label, score = get_tc_res(text)
    return {"code": 0, "msg": "success", "data": {"label": label, "score": score}}
