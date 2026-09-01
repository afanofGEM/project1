import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
'''规定日志的格式是 时间|日志等级|那个logger记录的|日志内容'''

logger = logging.getLogger('logger_predict')