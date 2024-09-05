import datetime
import math
import os
from functools import lru_cache
from logging import getLogger
from typing import List, Union

import numpy as np

# import tenbou
import uvicorn

# from algorithms import algorithms
# from cache import cache
from fastapi import Depends, FastAPI, Request, status  # , HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from andersan_backend import prefecture_retrievers

# class Predict(BaseModel):
#     loc: int = Field(example=14131030)
#     date: Union[datetime.datetime, List] # 時刻を複数与えてもいい。
#     algorithm: str = Field(example="mm_E")
#     fmax: int = Field(example=24)
#     col: str = Field(example="OX")


app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://172.23.78.218:8088",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/raw/{prefecture}/{isodate}")
async def raw_data(prefecture: str, isodate: str):
    """県内の全測定局の実測値を返す。

    Args:
        prefecture (str): 県名
        isodate (str): 時刻(isoformat)

    Returns:
        _type_: CSV形式(暫定)
    """
    assert prefecture in prefecture_retrievers
    # もうちょっと補助情報も出さないと使えないよ。
    return Response(
        content=prefecture_retrievers[prefecture].retrieve(isodate).to_csv()
    )


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8087,
        reload=True,
    )
