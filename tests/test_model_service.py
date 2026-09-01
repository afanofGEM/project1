import pytest
from app.model_service import encode_text, predict


'''mapping不是普通函数，而是一份可以提供给测试函数使用的“测试资源”
可以直接用函数名代表返回的char_to_id'''
@pytest.fixture
def mapping():
    return {
        "<PAD>": 0,
        "<UNK>": 1,
        "你": 2,
        "好": 3,
    }


def test_encode_text_adds_padding(mapping):
    result = encode_text("你好", mapping, max_len=4)

    assert result == [2, 3, 0, 0]


def test_encode_text_uses_unknown_token(mapping):
    result = encode_text("你呀", mapping, max_len=3)

    assert result == [2, 1, 0]


def test_encode_text_truncates_long_text(mapping):
    result = encode_text("你好你好", mapping, max_len=2)

    assert result == [2, 3]


def test_predict_rejects_unknown_model():
    '''我要盯着下面这一小段代码,如果它抛出 ValueError，并且错误信息 有 “Unknown model”，
    测试通过。'''
    with pytest.raises(ValueError, match="Unknown model"):
        predict("你好", "unknown")

