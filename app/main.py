from pathlib import Path
import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Query
from .model_service import predict
from .check import PredictRequest,PredictResponse,BatchPredictRequest,BatchPredictResponse
from time import perf_counter
from .database_session import create_session,save_prediction,save_predictions
from sqlmodel import Session
from .logging_config import logger
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test.csv"
BENCHMARK_MODELS = ("tfidf_lr", "mlp", "bert")

'''只要FastAPI遇到RequestValidationError，
不要使用FastAPI默认处理器,调用下面这个函数处理'''
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    '''取消RequestValidationError的处理逻辑
    改用logger记录warning和返回错误信息'''
    logger.warning(
        "Validation failed: path=%s errors=%s",
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request data",
            }
        },
    )


@app.get('/health')
def health():
    return {
        'status':'ok'
    }


'''predictrequest/response表示对请求和响应的数据格式是有要求的'''
@app.post('/predict',response_model=PredictResponse)
def predict_api(request:PredictRequest,
                session:Session = Depends(create_session)): 
    '''要赋值所以等号：先执行create_session()它会创建一个session,
    之后通过yield session把session传递到这个函数，这个过程就是依赖注入'''

    start_time = perf_counter()

    try:
      result = predict(request.text,request.model)
      '''{
          'model':'TF-IDF + LR',
          'predictions':predictions.item(),
          'max_probability':max_probability.item()
      }'''

    except Exception:
      logger.exception("Prediction failed: model=%s",
                      request.model)
      raise # 抛出异常

    end_time = perf_counter()
    latency_ms = (end_time - start_time) * 1000

    save_prediction(
        session=session,
        text=request.text,
        predicted_label=result["predictions"],
        confidence=result["max_probability"],
        model_name=result["model"],
        model_version="v1",
        latency_ms=latency_ms,
    )

    return result


@app.post('/batch_predict',response_model=BatchPredictResponse)
def batch_predict_api(request:BatchPredictRequest,
                session:Session = Depends(create_session)): 
    '''要赋值所以等号：先执行create_session()它会创建一个session,
    之后通过yield session把session传递到这个函数，这个过程就是依赖注入'''

    results = []
    model = request.model

    '''这里只是规定了logging的最后一项，内容'''
    logger.info("Batch prediction started: model_name=%s, texts_size=%d",
                model,len(request.text))
    
    for text in request.text:   
        start_time = perf_counter()

        try:
          result = predict(text,model)
          '''{
              'model':'TF-IDF + LR',
              'predictions':predictions.item(),
              'max_probability':max_probability.item()
          }'''

        except Exception:
          logger.exception("Prediction failed: model=%s text=%s",
                request.model,text)
          raise
          
        end_time = perf_counter()
        latency_ms = (end_time - start_time) * 1000

        save_prediction(
            session=session,
            text=text,
            predicted_label=result["predictions"],
            confidence=result["max_probability"],
            model_name=result["model"],
            model_version="v1",
            latency_ms=latency_ms,
        )  

        results.append(result)

    logger.info("Batch prediction ended: model_name=%s, texts_size=%d",
            model,len(request.text))
    
    return {
        'results':results
    }


@app.post("/benchmark/test-set")
# limit 即使不使用 Query 声明，FastAPI 也会把它识别为查询参数。
def benchmark_test_set(limit: int | None = Query(default=None,ge=1,
                                                 description="只评测前 N 条数据；不填写时评测整个测试集"),
                      session: Session = Depends(create_session)):

    """让三个模型分别预测测试集，并将每条预测批量写入数据库"""
    if not TEST_DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"找不到测试集：{TEST_DATA_PATH}",
        )

    test_data = pd.read_csv(TEST_DATA_PATH)

    if "text" not in test_data.columns or "label" not in test_data.columns:
        raise HTTPException(
            status_code=500,
            detail="测试集必须包含 text 和 label 字段",
        )

    if limit is not None:
        test_data = test_data.head(limit) # 不包括头部列名

    texts = test_data["text"].tolist()
    labels = test_data["label"].tolist()

    if len(texts) == 0:
        raise HTTPException(status_code=500, detail="测试集没有可评测的数据")

    summaries = {}
    database_records = []

    logger.info(
        "批量测试集数据测试开始: 样本数=%d 参与的基准模型=%s",
        len(texts),
        BENCHMARK_MODELS, 
    )

    for model_name in BENCHMARK_MODELS:

        '''懒加载模型并进行一次预热；这一段单独计时，不混入正式预测耗时。'''
        initialization_start = perf_counter()
        predict(texts[0], model_name)
        initialization_ms = (perf_counter() - initialization_start) * 1000

        correct = 0
        prediction_total_ms = 0.0

        for index in range(len(texts)):
            text = texts[index]
            true_label = labels[index]

            prediction_start = perf_counter()

            result = predict(text, model_name)
            '''{
                    'model':'MLP',
                    'predictions':id_to_label[labels_pred.item()],
                    'max_probability':max_probability.item()
                }'''
            
            latency_ms = (perf_counter() - prediction_start) * 1000

            prediction_total_ms += latency_ms
            if result["predictions"] == true_label:
                correct += 1

            database_records.append(
                {
                    "text": text,
                    "predicted_label": result["predictions"],
                    "confidence": result["max_probability"],
                    "model_name": result["model"],
                    "model_version": "v1",
                    "latency_ms": latency_ms,
                }
            )

        '''遍历样本结束'''
        sample_count = len(texts)
        summaries[model_name] = {
            "model": result["model"],
            "sample_count": sample_count,
            "correct_count": correct,
            "accuracy": correct / sample_count,
            "initialization_ms": initialization_ms,
            "prediction_total_ms": prediction_total_ms,
            "prediction_average_ms": prediction_total_ms / sample_count,
        }

        logger.info(
            "Model benchmark ended: model=%s samples=%d total_ms=%.3f accuracy=%.4f",
            model_name,
            sample_count,
            prediction_total_ms,
            correct / sample_count,
        )

    '''所有模型均测试完成'''
    save_predictions(session, database_records)

    logger.info(
        "Test-set benchmark ended: database_records=%d",
        len(database_records),
    )

    return {
        "test_data_path": str(TEST_DATA_PATH),
        "sample_count": len(texts),
        "database_records_created": len(database_records),
        "summaries": summaries,
    }

'''POST /predict
      ↓
    PredictRequest
      ↓
    pydantic开始检查request
    text 合法
    model_name 合法
      ↓
    Depends(get_session)
      ↓
    创建数据库 Session并依赖注入至本函数
      ↓
    开始计时
      ↓
    model_service.predict()
      ↓
    BERT
      ↓
    {
        "model": "bert",
        "predicted_label": "费用问题",
        "confidence": 0.9342
    }
      ↓
    计算 latency_ms
      ↓
    封装PredictionLog(...)
      ↓
    session.add()
      ↓
    session.commit()
      ↓
    MySQL prediction_logs +1 行
      ↓
    pydantic开始检查PredictResponse
      ↓
    PredictResponse'''
