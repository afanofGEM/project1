'''conftest.py是pytest里面的特殊类型文件
可以用于配置：
1. 临时环境变量'''

import os

'''os.environ表示环境变量的列表'''
os.environ.setdefault("DATABASE_URL","sqlite:///./pytest.db")
'''如果DATABASE_URL现在不存在，就设置成SQLite；如果它已经存在，就不要覆盖'''
