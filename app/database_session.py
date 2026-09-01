from sqlmodel import Session, create_engine
from .table import PredictionLog
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 读取 project1/.env
load_dotenv(PROJECT_ROOT / ".env")
# 从环境变量中读取数据库连接地址
DATABASE_URL = os.getenv("DATABASE_URL")

# 如果没有配置，立即停止，并显示清楚的错误信息
if not DATABASE_URL:
    raise RuntimeError(
        "没有找到 DATABASE_URL。"
        "请在 project1/.env 中配置数据库连接地址。"
    )

'''创建从程序到数据库的链接
它是整个FastAPI应用共享的'''
engine = create_engine(DATABASE_URL, echo=False,pool_pre_ping=True)

'''给每一次FastAPI请求临时创建一个数据库Session，用完以后自动关闭'''
def create_session():
    with Session(engine) as session:
        yield session
        
'''Session(engine):基于这个engine，创建一个数据库会话
        ↓
    进入 with
        ↓
    创建 session
        ↓
    yield session把session借给别的程序使用,操作数据库
        ↓
    回到yield session,离开 with
        ↓
    自动关闭 session'''


'''主函数，定义数据库存储操作的'''
def save_prediction(session: Session,text: str,predicted_label: str,
                    confidence: float,model_name: str,model_version: str,
                    latency_ms: float):
    
    log = PredictionLog(text=text,
                        predicted_label=predicted_label,
                        confidence=confidence,
                        model_name=model_name,
                        model_version=model_version,
                        latency_ms=latency_ms,
                    )
    '''主键和时间都是默认值，自增和当前时间'''

    try:
        session.add(log)
        '''session:Session(engine)它是操作这个数据库的session，现在需要定义这是什么操作
        add(log)就是执行add操作'''

        session.commit()
        '''定义好操作类型之后就可以执行了'''

    except Exception:
        session.rollback()
        '''如果执行失败了就回滚(ctrl+z)'''

        '''抛异常'''
        raise


def save_predictions(session: Session, predictions: list[dict]):
    '''predictions:
    [{
        "text": text,
        "predicted_label": result["predictions"],
        "confidence": result["max_probability"],
        "model_name": result["model"],
        "model_version": "v1",
        "latency_ms": latency_ms,
    },...]'''

    """**prediction 就是把字典拆成关键字参数"""
    logs = [PredictionLog(**prediction) for prediction in predictions]

    try:
        session.add_all(logs)
        session.commit()
    except Exception:
        session.rollback()
        raise
