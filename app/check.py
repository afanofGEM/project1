from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path
import json
from typing import Annotated

project_path = Path(__file__).parent.parent
outputs_root = project_path / 'outputs' 
max_len_path = outputs_root / 'decide_max_len' / 'max_len.json'
with max_len_path.open('r',encoding='utf-8') as file:
    max_len = json.load(file)['max_len']

'''不满足限制的会抛异常RequestValidationError，错误码422'''

'''BaseModel表示此类是用pydantic监测数据规范的'''
class PredictRequest(BaseModel):
    text:str = Field(...,min_length=1,max_length=max_len) 
    '''text类型str,
    需要满足: ... 必填字段  长度在1-max_len之间'''

    model:Literal['tfidf_lr','mlp','bert'] = 'tfidf_lr'
    '''model: 值必须是3个里面选一个，而且默认值是tfidf_lr'''


class PredictResponse(BaseModel):
    model:str
    predictions:str
    max_probability:float = Field(...,ge=0,le=1)
    '''ge:equal or greater than
    le: equal or less than'''

'''Annotated给一个已有类型附加额外规则'''
Text = Annotated[str,Field(...,min_length=1,max_length=max_len)]

'''BaseModel表示此类是用pydantic监测数据规范的'''
class BatchPredictRequest(BaseModel):

    '''此处的max/min_length是规定一共可以输入多少条文本'''
    text:list[Text] = Field(...,min_length=1,max_length=max_len) 
    model:Literal['tfidf_lr','mlp','bert'] = 'tfidf_lr'


class BatchPredictResponse(BaseModel):
    results:list[PredictResponse]

