from datetime import datetime
from sqlmodel import Field, SQLModel

'''代表这是一个真正对应数据库表的table model，而一个实例可以代表表中的一行'''
class PredictionLog(SQLModel, table=True):

    __tablename__ = "prediction_logs"

    '''主键可以不填，None，默认值就是None因为主键自增'''
    id: int | None = Field(default=None, primary_key=True)

    text: str = Field(...,max_length=200)
    predicted_label: str = Field(...,max_length=50)
    confidence: float = Field(...)
    model_name: str = Field(...,max_length=50)
    model_version: str = Field(...,max_length=50)
    latency_ms: float = Field(...)
    '''默认值就是当前时间'''
    created_at: datetime = Field(...,default_factory=datetime.now)


'''ORM：python Object -> table Row'''
