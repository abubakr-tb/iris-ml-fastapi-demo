from array import array
import numpy as np
from fastapi import FastAPI, Form, Body
from pydantic import BaseModel
import ml
import pandas as pd
from typing import List, Union
from fastapi.encoders import jsonable_encoder


app = FastAPI()

class Input(BaseModel):
    SepalLength: float
    SepalWidth: float
    PetalLength: float
    PetalWidth: float

class item(BaseModel):
    items: List[Input]

# class Response(BaseModel):
#     loss: float
#     accuracy: float
#     actual: List[str]=[]
#     predicted: List[str]=[]

@app.get("/")
async def root():
    return {"message": "Such ML much wow"}

@app.post("/predictions")
async def prediction(input: dict = Body(...)):
    
    ml.load_file()

    # allSepalLength, allSepalWidth, allPetalLength, allPetalWidth =[],[],[],[]
    # print(type(storage))
    # print(input)
    body = input['items']
    # print(body)
    print([np.array(item.values()) for item in body])

    # print(body)

    [
        [5.1, 3.5, 1.4, 0.2],
        [5.1, 3.5, 1.4, 0.2]
    ]


    return body
    # data = pd.DataFrame.from_dict(body)
    # data.head()

    # for things in body:
    #     data = pd.DataFrame.from_dict(things)
    # print(data)
        # data = things.items()
        # listy = list(data)
        # arr = np.array(listy)
        # print(arr)

    # X = data.iloc[:,0:4].values
    # y = data.iloc[:,1].values
    # print(X[0:5])


    # print(y[0:5])
    # y = ml.encoder_transform(Y)

    # final = ml.train_test(X,y)

    # return input
    #     allSepalLength.append(things['SepalLength'])
    #     allSepalWidth.append(things['SepalWidth'])
    #     allPetalLength.append(things['PetalLength'])
    #     allPetalWidth.append(things['PetalWidth'])
    #     np1=np.array(allSepalLength)
    #     np2=np.array(allSepalWidth)
    #     np3=np.array(allPetalLength)
    #     np4=np.array(allPetalWidth)
    #     df = pd.DataFrame(data=[np1, np2, np3, np4]).T
    # # print(df)
    
    # print(np1)
    # print(np2)
    # print(np3)
    # print(np4)
            
    # print(allSepalLength)
    # print(allSepalWidth)
    # print(allPetalLength)
    # print(allPetalWidth)
    # for stuffs in storage:
    #     # print(stuffs[1])
    #     for things in stuffs:
    #         # print(things)
    #         for goods in things:
    #             print(goods)
    # print(json_item_data)
    # print(type(input))
    # return input
