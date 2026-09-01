from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.database_session import create_session
from app.main import app


def override_session():
    yield Mock()

'''对fastapi服务器进行设置，替换函数'''
app.dependency_overrides[create_session] = override_session # 不要写入真实数据库

'''创造一个虚拟的用户'''
client = TestClient(app)


def test_health():
    response = client.get("/health")
    '''相当于GET /health，它比直接执行def health()的流程更加完整'''

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


'''测试/predict本身有没有正确接收请求、调用业务函数predict和save_prediction、返回结果'''
@patch("app.main.save_prediction")
@patch("app.main.predict")
def test_predict_api(mock_predict,mock_save_prediction):
    '''把原先的两个函数替换成这两个函数'''
    mock_predict.return_value = {
        "model": "TF-IDF + LR",
        "predictions": "weather",
        "max_probability": 0.95,
    }

    '''不设置save_prediction的return_value,就是啥也不干'''
    response = client.post('/predict', json={
                                            "text": "今天会下雨吗",
                                            "model": "tfidf_lr",
                                        })

    '''检查 HTTP 是否成功'''
    assert response.status_code == 200

    '''检查 API 最终返回内容是否就是预测的内容，而不是其他乱七八糟的'''
    assert response.json()["predictions"] == "weather"
    assert response.json()["max_probability"] == 0.95

    '''检查数据库写入函数是否只调用了一次，而且text,model
    是否被传递给了保存函数'''
    mock_save_prediction.assert_called_once()
    saved = mock_save_prediction.call_args.kwargs
    assert saved["text"] == "今天会下雨吗"
    assert saved["predicted_label"] == "weather"
    assert saved["model_name"] == "TF-IDF + LR"


'''当传递的是空文本时，测试错误相应内容
与@app.exception_handler(RequestValidationError)对应'''
def test_rejects_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": "",
            "model": "tfidf_lr",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["message"] == "Invalid request data"


def test_predict_rejects_invalid_model():
    response = client.post(
        "/predict",
        json={
            "text": "测试文本",
            "model": "invalid",
        },
    )
    
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["message"] == "Invalid request data"
