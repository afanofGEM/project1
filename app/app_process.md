1. 收到post请求，路由是"/predict"
@app.post('/predict')

2. **Router层**函数规定post请求和response各个参数的 
数据类型，是否为空，默认值，备选值等
@app.post('/predict',response_model=PredictResponse)
def predict_api(request:PredictRequest): 

PredictResponse,PredictRequest中：
- 冒号:后跟着数据类型
- Field中包括着参数需要满足的条件：
    - ... 表示不为空
    - min,max
    - Literal规定取值范围
    - ge le: 小于等于，大于等于

3. 因为所调用函数涉及到数据库操作，需要为数据库操作创建一个Session
session:Session = Depends(create_session)
**DAO层**
执行'''create_session'''自定义函数

- 先创建py程序与数据库的连接engine
- 利用engine创建操作数据库对应表的session
- 通过yield把engine创建的session租借给main函数

4. 调用**Service层**的predict函数：
接收请求中的参数，返回结果参数dict

5. 调**DAO层**的保存至数据库函数save_prediction：

- 通过设置class PredictionLog(SQLModel, table=True)告诉
  这个对象对应着数据库表的行
  ORM：python Object -> table Row
  **ORM model**

- 再通过session把对象写入表，实现ORM