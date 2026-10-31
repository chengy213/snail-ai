## 一、SQLAlchemy 是什么

SQLAlchemy 是 Python 中最流行的数据库工具库，它让开发者可以用 Python 代码代替 SQL 语句来操作数据库。
它分为两层：底层叫 Core，负责连接池管理和 SQL 语句构建，适合需要精细控制 SQL 的场景；上层叫 ORM，能将数据库表映射成 Python 类，让你像操作普通对象一样增删改查，无需手写 SQL。简单说，它把数据库的行变成了对象，把列变成了属性。

## 二、安装与数据库连接

安装只需一行 pip install sqlalchemy，但连接不同数据库还需要对应的驱动。
连接的核心是创建引擎，它通过一个 URL 来指定数据库地址、账号密码和目标库。引擎内部维护着一个连接池，复用连接能显著提升性能。除了引擎，你还需要一个会话工厂，每次数据库操作都应该创建一个新的会话，用完即关，避免数据混乱。

```python
# 连接的核心是创建引擎，它通过一个 URL 来指定数据库地址、账号密码和目标库。
# 引擎内部维护着一个连接池，复用连接能显著提升性能。
# 除了引擎，你还需要一个会话工厂，每次数据库操作都应该创建一个新的会话，用完即关，避免数据混乱。
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/snail-cloud?charset=utf8mb4"

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    echo=True,  # 打印 SQL，调试用
    pool_size=10,  # 连接池大小
    max_overflow=5,  # 最大溢出连接数
    pool_recycle=3600,  # 连接回收时间
    pool_pre_ping=True  # 检查连接是否可用
)

# 创建会话工厂
SessionLocal  = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)
```

## 三、定义 ORM 模型

模型就是一个 Python 类，用来描述一张表的结构：类名对应表名，属性对应列。
首先创建一个基类，然后让所有模型继承它。每个属性用类型注解标明它存什么数据，比如 int 还是 str，同时可以加上约束：主键、唯一、索引、默认值等。定义好模型后，数据库表的结构就被清晰地表达在代码中了，这是后续一切操作的基础。

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, func
from datetime import datetime
# ============================================================
# 第一步：创建基类
# 所有模型类都继承自它，SQLAlchemy 通过它识别哪些是 ORM 模型
# ============================================================
class Base(DeclarativeBase):
    pass
# ============================================================
# 第二步：定义 User 模型——映射到 users 表
# ============================================================
class User(Base):
    # 表名
    __tablename__ = "users"
    # 主键，自增，唯一标识每一行
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # 用户名，最长50字符，不可重复，加索引加速查询
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    # 邮箱，最长100字符，不可重复
    email: Mapped[str] = mapped_column(String(100), unique=True)
    # 年龄，整数，默认值为 0
    age: Mapped[int] = mapped_column(Integer, default=0)
    # 个人简介，长文本，可选（允许为空）
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 创建时间，插入行时自动填入当前时间
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # 更新时间，每次修改行时自动更新为当前时间
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),   # 首次插入时的默认值
        onupdate=func.now()   # 更新数据时的自动值
    )
    # ----------------------------------------------------------
    # 关联关系：User 与 Post 是一对多（一个用户有多篇文章）
    # 这不是数据库中的真实列，而是 ORM 层面的对象引用
    # back_populates 指向 Post 模型中对应关系的属性名
    # ----------------------------------------------------------
    posts: Mapped[list["Post"]] = relationship(back_populates="author")


# ============================================================
# 第三步：定义 Post 模型——映射到 posts 表
# ============================================================
class Post(Base):
    # 表名
    __tablename__ = "posts"

    # 主键，自增
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 文章标题，最长200字符
    title: Mapped[str] = mapped_column(String(200))

    # 文章内容，长文本
    content: Mapped[str] = mapped_column(Text)

    # 外键，指向 users 表的 id 列，表名该文章属于哪个用户
    # 加索引，方便根据用户查文章时加速
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    # ----------------------------------------------------------
    # 反向关联关系：通过 post.author 可直接拿到对应的 User 对象
    # ----------------------------------------------------------
    author: Mapped["User"] = relationship(back_populates="posts")
```

## 四、自动创建数据表

模型定义好之后，调用基类的 create_all 方法，SQLAlchemy 会自动检查数据库中哪些表还不存在，并帮你创建。非常适合开发阶段快速建表。但生产环境通常会改用专业的迁移工具（如 Alembic）来精细控制表结构的变更历史。

```python
# ============================================================
# 自动建表——开发阶段用
# 检查数据库中哪些表不存在，只创建缺失的表，已存在的不会动
# 注意：如果表结构变了（比如加了列），它不会自动更新，需要删表重建
# ============================================================
Base.metadata.create_all(bind=engine)
```

## 五、数据库会话依赖注入

在 Web 应用中，每个请求都应该拥有独立的数据库会话，请求结束时就关闭它，避免连接泄漏和数据串扰。
借助 FastAPI 的依赖注入机制，可以把“获取会话”写成一个可复用的函数。在路由函数中声明依赖后，框架会自动在请求到来时创建会话，在请求处理完毕后关闭它。这样路由函数里只需关注业务逻辑，不用关心会话的创建和销毁。

```python
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# ============================================================
# 全局：引擎和会话工厂（只创建一次）
# ============================================================
DATABASE_URL = "mysql+pymysql://root:password@localhost:3306/mydb?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
# ============================================================
# 依赖函数：每个请求获取一个独立会话
# 用 yield 实现：请求进来时创建会话，请求结束时自动关闭
# ============================================================
def get_db():
    """FastAPI 依赖注入，为每个请求提供独立的数据库会话"""
    db = SessionLocal()     # 请求进来，创建新会话
    try:
        yield db            # 把会话交给路由函数使用
    finally:
        db.close()          # 请求结束，关闭会话，归还连接

# ============================================================
# 创建 FastAPI 应用
# ============================================================
app = FastAPI()
# ============================================================
# 路由函数：只需声明 db 依赖，框架自动注入
# 业务代码里直接用 db，不用关心会话生命周期
# ============================================================
@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    """查询单个用户"""
    user = db.query(User).filter(User.id == user_id).first()
    return user

@app.post("/users")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    """新增用户"""
    user = User(name=name, email=email)
    db.add(user)
    
    db.commit()
    db.refresh(user)
    return user
```

## 六、基础增删改查操作

这是最常用的四项操作，都通过会话对象来执行。
增：先创建模型实例，然后用 add 加入会话，最后 commit 提交。提交后可用 refresh 获取数据库生成的默认值（如自增 ID）。
查：用 query 链式调用 filter 指定条件，再调用 first() 取第一条或 all() 取所有。
改：先查到要修改的对象，直接修改它的属性，再 commit 即可。ORM 会自动追踪属性的变化。
删：查到对象后，用 delete 标记删除，然后 commit 提交。10
核心思路是：一切操作先发生在 Python 对象上，commit 时才同步到数据库。

```python
# ============================================================
# 增删改查 —— 全部通过会话对象 db 执行
# ============================================================


# ============================================================
# 一、增（Create）
# 1. 创建模型实例（此时只在 Python 内存中，未进数据库）
# 2. add 加入会话的待提交队列
# 3. commit 真正写入数据库
# 4. refresh 从数据库回读，拿到自增 ID 等数据库生成的值
# ============================================================
new_user = User(username="张三", email="zhangsan@test.com", age=25)
db.add(new_user)        # 加入待提交队列，还没写入数据库
db.commit()             # 真正执行 INSERT 语句，写入数据库
db.refresh(new_user)    # 回读数据库生成的值，比如 new_user.id 现在是自增后的 ID


# ============================================================
# 二、查（Read）
# query(模型类) 指定查哪张表
# filter() 加条件过滤
# first() 取第一条，查不到返回 None
# all() 取所有，返回列表
# ============================================================
# 查单条
user = db.query(User).filter(User.email == "zhangsan@test.com").first()

# 查所有
users = db.query(User).filter(User.age > 18).all()

# 按主键查（更快的简写）
user = db.query(User).get(1)


# ============================================================
# 三、改（Update）
# 1. 先查到要改的对象
# 2. 直接修改属性值（ORM 会自动追踪变化）
# 3. commit 生成 UPDATE 语句同步到数据库
# 无需手动调用 update 方法
# ============================================================
user = db.query(User).filter(User.id == 1).first()
if user:
    user.age = 30          # 直接改属性
    user.email = "new@test.com"
    db.commit()            # ORM 自动对比变化，生成 UPDATE 语句


# ============================================================
# 四、删（Delete）
# 1. 先查到要删的对象
# 2. delete 标记为待删除
# 3. commit 执行 DELETE 语句
# ============================================================
user = db.query(User).filter(User.id == 1).first()
if user:
    db.delete(user)        # 标记删除
    db.commit()            # 真正从数据库删除


# ============================================================
# 核心流程总结：
# Python 对象（内存） → add / 改属性 / delete → commit → 数据库
# commit 之前，一切变化只在内存中，不影响数据库
# 如果中途异常，没 commit 的改动会自动丢弃，天然安全
# ============================================================
```

## 七、复杂条件查询

除了简单的等值过滤，SQLAlchemy 还支持丰富的查询表达。
你可以组合多个条件，比如 “年龄大于 18 且城市是北京”；也可以做模糊匹配，比如 “名字中含‘张’”；还能进行聚合统计，比如 “每个城市各有多少用户”。对于多表查询，如果模型之间定义了关联关系，可以直接用 join 跨表过滤，就像操作单个对象一样自然。

```python
from sqlalchemy import or_, and_, func, between

# ============================================================
# 一、多条件组合
# ============================================================

# AND 连接：年龄大于18 且 城市是北京
users = db.query(User).filter(
    and_(User.age > 18, User.city == "北京")
).all()

# OR 连接：城市是北京 或 上海
users = db.query(User).filter(
    or_(User.city == "北京", User.city == "上海")
).all()

# 链式 filter 也等同于 AND
users = db.query(User).filter(User.age > 18).filter(User.city == "北京").all()

# 区间查询：年龄在 20 到 30 之间
users = db.query(User).filter(between(User.age, 20, 30)).all()

# IN 查询：城市在给定列表中
users = db.query(User).filter(User.city.in_(["北京", "上海", "广州"])).all()


# ============================================================
# 二、模糊匹配
# ============================================================

# 名字中含"张"
users = db.query(User).filter(User.name.like("%张%")).all()

# 名字以"张"开头
users = db.query(User).filter(User.name.like("张%")).all()

# 邮箱以 @gmail.com 结尾
users = db.query(User).filter(User.email.like("%@gmail.com")).all()


# ============================================================
# 三、聚合统计
# ============================================================

# 统计用户总数
count = db.query(func.count(User.id)).scalar()

# 按城市分组，统计每个城市的用户数
result = db.query(User.city, func.count(User.id)).group_by(User.city).all()
# 返回：[("北京", 15), ("上海", 12), ("广州", 8)]

# 平均年龄
avg_age = db.query(func.avg(User.age)).scalar()


# ============================================================
# 四、多表关联查询
# 前提：User 和 Post 已通过 relationship 和 ForeignKey 建立关联
# ============================================================

# 方式一：join + filter，跨表过滤
# 查所有"张三"写的文章
posts = db.query(Post).join(User).filter(User.name == "张三").all()

# 方式二：直接用 relationship 属性，更自然
# 先拿到用户，再通过 .posts 拿到他的所有文章
user = db.query(User).filter(User.name == "张三").first()
posts = user.posts   # 这就是他写的所有文章

# 反向查：这篇文章的作者是谁
post = db.query(Post).filter(Post.title == "某文章标题").first()
author = post.author  # 直接拿到 User 对象
```

## 八、原生 SQL 执行

当 ORM 的表达能力不够用时，可以直接写 SQL。SQLAlchemy 提供了 text() 函数来包裹原生 SQL 字符串，并通过字典方式传入参数，能有效防止 SQL 注入。这种方法保留了 ORM 的连接管理和事务控制，同时让你在复杂场景下直接掌控 SQL 逻辑，是 ORM 能力的有力补充。

```python
from sqlalchemy import text

# ============================================================
# 原生 SQL 执行 —— text() 包裹 SQL，字典传参防注入
# ============================================================
# ============================================================
# 一、查询操作
# ============================================================

# 简单查询
sql = "SELECT * FROM users WHERE age > :age"
rows = db.execute(text(sql), {"age": 18}).fetchall()
# rows 是列表，每行是一个 Row 对象，可按列名取值

# 遍历结果
for row in rows:
    print(row.id, row.name, row.email)


# 取单条
sql = "SELECT * FROM users WHERE id = :user_id"
row = db.execute(text(sql), {"user_id": 1}).fetchone()


# 多条件参数化
sql = """
    SELECT * FROM users 
    WHERE age > :min_age 
    AND city = :city
"""
rows = db.execute(text(sql), {"min_age": 20, "city": "北京"}).fetchall()


# 聚合查询
sql = "SELECT city, COUNT(*) as cnt FROM users GROUP BY city"
rows = db.execute(text(sql)).fetchall()
for row in rows:
    print(row.city, row.cnt)


# ============================================================
# 二、写操作（增删改）
# 注意：execute 后必须手动 commit，和 ORM 方式一样
# ============================================================

# 插入
sql = "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)"
db.execute(text(sql), {"name": "王五", "email": "wangwu@test.com", "age": 28})
db.commit()   # 必须提交

# 更新
sql = "UPDATE users SET age = :age WHERE id = :user_id"
db.execute(text(sql), {"age": 30, "user_id": 1})
db.commit()

# 删除
sql = "DELETE FROM users WHERE id = :user_id"
db.execute(text(sql), {"user_id": 5})
db.commit()

# ============================================================
# 三、复杂查询示例（ORM 难表达的 SQL）
# ============================================================

# 子查询
sql = """
    SELECT * FROM users 
    WHERE id IN (
        SELECT user_id FROM posts 
        WHERE created_at > :since
    )
"""
rows = db.execute(text(sql), {"since": "2025-01-01"}).fetchall()

# 联表查询
sql = """
    SELECT u.name, p.title, p.created_at
    FROM users u
    JOIN posts p ON u.id = p.user_id
    WHERE u.city = :city
    ORDER BY p.created_at DESC
"""
rows = db.execute(text(sql), {"city": "北京"}).fetchall()
```

## 九、事务管理

事务能保证一组操作要么全部成功，要么全部撤销，这是确保数据一致性的关键。
SQLAlchemy 默认在 commit 时提交事务。如果中间发生异常，需要手动调用 rollback 回滚。更推荐的方式是使用 with 语句开启事务，代码块结束时自动提交，有异常则自动回滚，代码更简洁也更安全。

```python
# ============================================================
# 事务管理 —— 保证一组操作同生共死
# ============================================================

# ============================================================
# 场景示例：用户下单
# 需要同时做两件事：1. 扣减库存  2. 创建订单
# 任何一步失败，另一条也必须撤销，不能出现库存扣了但订单没生成
# ============================================================

# ============================================================
# 方式一：手动提交 / 回滚（原始写法）
# ============================================================
def create_order_manual(user_id, product_id, quantity):
    db = SessionLocal()
    try:
        # 步骤1：扣减库存
        product = db.query(Product).filter(Product.id == product_id).first()
        product.stock -= quantity

        # 步骤2：创建订单
        order = Order(user_id=user_id, product_id=product_id, quantity=quantity)
        db.add(order)

        # 都成功，提交
        db.commit()
    except Exception:
        # 任何一步出错，回滚所有改动
        db.rollback()
        raise   # 把异常继续往上抛
    finally:
        db.close()


# ============================================================
# 方式二：with 语句自动管理（推荐）
# session.begin() 开启事务
# 代码块正常结束 → 自动 commit
# 代码块抛出异常 → 自动 rollback
# 不用手写 try/except/finally
# ============================================================
def create_order_with(user_id, product_id, quantity):
    with SessionLocal() as db:               # 自动关闭会话
        with db.begin():                     # 开启事务，自动提交或回滚
            # 步骤1：扣减库存
            product = db.query(Product).filter(Product.id == product_id).first()
            product.stock -= quantity

            # 步骤2：创建订单
            order = Order(user_id=user_id, product_id=product_id, quantity=quantity)
            db.add(order)

        # 到这里事务已自动提交，无需手动 commit


# ============================================================
# 方式三：更简洁的嵌套 with（一个 with 同时管会话和事务）
# ============================================================
def create_order_compact(user_id, product_id, quantity):
    with SessionLocal() as db, db.begin():   # 逗号并列，一个缩进层级
        # 扣库存
        product = db.query(Product).filter(Product.id == product_id).first()
        product.stock -= quantity

        # 建订单
        order = Order(user_id=user_id, product_id=product_id, quantity=quantity)
        db.add(order)
    # 离开 with 块时，自动提交并关闭会话


# ============================================================
# 对比说明
# ============================================================
```

| 写法                              | 优点                       | 缺点                                 |
| :-------------------------------- | :------------------------- | :----------------------------------- |
| 手动 `try/except`                 | 流程清晰，可自定义异常处理 | 代码啰嗦，容易忘写 rollback 或 close |
| `with db.begin()`                 | 异常自动回滚，不会忘       | 需要额外嵌一层 with                  |
| `with SessionLocal(), db.begin()` | 一行搞定，最简洁           | 无法在中间插入自定义异常处理         |