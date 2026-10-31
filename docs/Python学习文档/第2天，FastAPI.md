# FastAPI

## 1、FastAPI 是什么

它是 Python 现代高性能 Web 框架，对标 Java 的 SpringBoot + SpringMVC。优点是速度快、自动生成接口文档、支持异步、自带类型校验、代码极度简洁，现在后端开发、AI 接口服务、微服务项目几乎都在用。

## 2、安装与启动

Java 用 Maven 或 Gradle 引入依赖；Python 只需要用 pip 安装，启动命令也非常简单，支持热重载，改代码自动刷新，开发效率很高。

**Java（Maven）**

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

**Python（FastAPI）**

```java
pip install fastapi uvicorn
uvicorn main:app --reload
```

## 3、请求流程

和 SpringBoot 的请求处理流程几乎一致：请求到达 → 路由匹配 → 参数解析与校验 → 执行业务逻辑 → 返回响应。FastAPI 基于协程处理请求，高并发场景下性能更强。

**Java Spring MVC**：

```
Filter → DispatcherServlet → HandlerMapping → Controller → ViewResolver → 响应
```

**FastAPI**：

```
Middleware → Router匹配 → 参数解析(Pydantic) → 依赖注入(Depends) → 路由函数 → 响应
```

```python
from fastapi import FastAPI, Request

app = FastAPI()

# 中间件：请求进来最先执行，最后返回
@app.middleware("http")
async def log_middleware(request: Request, call_next):
    print(f"1. 中间件前置: {request.method} {request.url}")
    response = await call_next(request)  # 进入路由处理
    print(f"4. 中间件后置: {response.status_code}")
    return response

# 依赖注入：中间件之后、路由函数之前执行
async def get_db():
    print("2. 依赖注入: 获取数据库连接")
    yield "db_connection"
    print("5. 依赖清理: 释放数据库连接")

# 路由函数
@app.get("/hello")
async def hello(name: str, db: str = Depends(get_db)):
    print(f"3. 路由处理: name={name}, db={db}")
    return {"message": f"Hello {name}"}
```

```txt
执行顺序：
1. 中间件前置: GET /hello?name=world
2. 依赖注入: 获取数据库连接
3. 路由处理: name=world, db=db_connection
4. 中间件后置: 200
5. 依赖清理: 释放数据库连接
```

## 4、路由分层

对应 SpringBoot 的 Controller 分包，FastAPI 支持路由拆分，不同业务模块分开管理，大型项目结构清晰、易于维护。

**Java（Controller）**

```java
@RestController
@RequestMapping("/user")
public class UserController {
    @GetMapping("/list")
    public List<User> list() {
        return List.of(new User(1), new User(2));
    }
}
```

**Python（APIRouter）**

```python
from fastapi import APIRouter
user_router = APIRouter(prefix="/user", tags=["用户模块"])

@user_router.get("/list")
def user_list():
    return [{"id": 1}, {"id": 2}]
```

## 5、查询参数、请求体

对应 SpringBoot 的 @RequestParam和@RequestBody，FastAPI 直接定义参数和用 Pydantic 接收 JSON，自带类型校验、默认值

**Java**

```java
@GetMapping("/search")
public String search(@RequestParam String keyword, @RequestParam(defaultValue = "1") int page) {
    return keyword;
}

public class User {
    private String name;
    private Integer age;
}

@PostMapping("/user")
public User create(@RequestBody User user) {
    return user;
}
```

**Python**

```python
@app.get("/search")
def search(keyword: str, page: int = 1):
    return {"keyword": keyword, "page": page}

from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

@app.post("/user")
def create_user(user: User):
    return {"msg": "创建成功", "data": user}
```

## 6、Pydantic 数据校验

。对标 Java 的参数校验框架，支持非空、长度、范围、格式等多种校验规则，接口更安全、更规范。

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
import re

# ===== 基础字段约束 =====
class ProductCreate(BaseModel):
    name: str = Field(
        ...,  # ... 表示必填
        min_length=1,
        max_length=100,
        description="商品名称",
        examples=["iPhone 15"]
    )
    price: float = Field(
        ...,
        gt=0,  # 大于0
        le=999999.99,  # 小于等于
        description="价格"
    )
    stock: int = Field(
        default=0,
        ge=0,  # 大于等于0
        description="库存"
    )
    tags: List[str] = Field(
        default_factory=list,
        max_length=10,  # 列表最多10个元素
    )
    metadata: Dict[str, str] = Field(default_factory=dict)
    
    # 枚举字段
    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"
        ARCHIVED = "archived"
    
    status: Status = Field(default=Status.DRAFT)
    
    # 日期时间
    publish_at: Optional[datetime] = None


# ===== 字段级校验 =====
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=8)
    confirm_password: str
    email: str
    phone: str
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """用户名只能包含字母数字下划线"""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("用户名只能包含字母、数字和下划线")
        return v.strip()
    
    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """手机号格式校验"""
        if not re.match(r"^1[3-9]\d{9}$", v):
            raise ValueError("手机号格式不正确")
        return v
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v:
            raise ValueError("邮箱格式不正确")
        return v


# ===== 跨字段校验 =====
class PasswordChange(BaseModel):
    old_password: str
    new_password: str
    confirm_new_password: str
    
    @field_validator("confirm_new_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """确认密码必须和新密码一致"""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("两次输入的新密码不一致")
        return v
    
    @field_validator("new_password")
    @classmethod
    def different_from_old(cls, v: str, info) -> str:
        if "old_password" in info.data and v == info.data["old_password"]:
            raise ValueError("新密码不能与旧密码相同")
        return v


# ===== 模型级校验（所有字段验证后） =====
class OrderCreate(BaseModel):
    items: List[str]
    coupon_code: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_order(self):
        """订单级别校验"""
        if len(self.items) == 0:
            raise ValueError("订单必须包含至少一个商品")
        
        if self.coupon_code and len(self.items) < 2:
            raise ValueError("优惠券需要至少2件商品才能使用")
        
        return self


# ===== 常见约束速查 =====
"""
字符串:
    min_length      - 最小长度
    max_length      - 最大长度
    pattern         - 正则表达式

数字:
    gt              - 大于
    ge              - 大于等于
    lt              - 小于
    le              - 小于等于
    multiple_of     - 倍数

通用:
    default         - 默认值
    default_factory - 默认值工厂函数(适合[]、{})
    description     - 字段描述(生成到文档)
    examples        - 示例值(生成到文档)
    deprecated      - 标记弃用
    alias           - 字段别名(如JSON的camelCase)
"""
```

## 7、自动生成接口文档

SpringBoot 需要集成 Swagger；FastAPI 零配置自带文档，相当于开箱即用的 Swagger。

```python
from fastapi import FastAPI

app = FastAPI(
    title="我的API",
    version="1.0.0",
    description="这是API描述"
)

@app.get("/hello")
async def hello(name: str = "World"):
    return {"message": f"Hello {name}"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

```txt
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

## 8、依赖注入

对应 SpringBoot @Autowired，FastAPI 使用 Depends，用于数据库连接、权限、公共逻辑。

**Java**

```java
@Service
public class UserService {}

@RestController
public class UserController {
    @Autowired
    private UserService userService;
}
```

**Python**

```python
def get_db():
    db = "数据库连接"
    yield db
    db = "关闭连接"

@app.get("/data")
def get_data(db = Depends(get_db)):
    return {"db": db}
```

## 9、AOP 与中间件

对应 Spring AOP / 拦截器，FastAPI 中间件做全局日志、跨域、请求头、异常处理。

**Java（拦截器）**

```java
public class MyInterceptor implements HandlerInterceptor {
    // 前置/后置处理
}
```

**Python**

```python
@app.middleware("http")
async def my_middleware(request, call_next):
    # 前置处理
    response = await call_next(request)
    # 后置处理
    return response
```

## 10、异步支持

对应 Java CompletableFuture / WebFlux，FastAPI 原生 async/await，IO 密集型接口性能极强。

**Java**

```java
@GetMapping("/async")
public CompletableFuture<String> async() {
    return CompletableFuture.supplyAsync(() -> "异步结果");
}
```

**Python**

```python
@app.get("/async")
async def async_api():
    return {"msg": "异步接口"}
```

## 11、ORM 数据库操作

对应 Java MyBatis / JPA，FastAPI 搭配 SQLAlchemy / Tortoise，实现对象映射、增删改查。

**Java（JPA）**

```java
@Entity
public class User {
    @Id
    private Long id;
    private String name;
}
```

**Python（SQLAlchemy）**

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
```