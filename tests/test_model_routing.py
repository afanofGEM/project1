'''测试推理分发，但不加载真实模型'''
from unittest.mock import patch
from app.model_service import predict

'''测试期间，暂时把真正的predict_baseline
替换成一个假的函数，也就是mock_predict。
于是原本predict_baseline(...)不会真的执行模型预测'''
@patch("app.model_service.predict_baseline")
def test_predict_routes_to_baseline(mock_predict):

    mock_predict.return_value = {
        "model": "TF-IDF + LR",
        "predictions": "weather",
        "max_probability": 0.9,
    } # 给这个假的mock函数固定一个返回值

    result = predict("今天天气怎么样", "tfidf_lr")
    '''理论上应该走predict_baseline(text)
    但是因为@patch("app.model_service.predict_baseline")
    所以实际上调用的是：mock_predict("今天天气怎么样")
    于是得到刚才人为规定的：
    {
        "model": "TF-IDF + LR",
        "predictions": "weather",
        "max_probability": 0.9,
    }'''

    mock_predict.assert_called_once_with("今天天气怎么样")
    '''检查predict_baseline 是否恰好被调用了一次，而且参数是不是 "今天天气怎么样"'''

    assert result["predictions"] == "weather"


@patch("app.model_service.predict_mlp")
def test_predict_routes_to_mlp(mock_predict):
    mock_predict.return_value = {
        "model": "MLP",
        "predictions": "weather",
        "max_probability": 0.8,
    }

    '''predict调predict_mlp,predict_mlp的结果被人为规定'''
    result = predict("今天天气怎么样", "mlp")

    mock_predict.assert_called_once_with("今天天气怎么样")
    '''验证predict函数是否能正确执行一次predict_mlp'''  

    assert result["model"] == "MLP" 
    '''验证predict函数是否能正确返回predict_mlp结果'''
