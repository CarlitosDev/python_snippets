from fastapi import FastAPI

app = FastAPI()

@app.get("/students/{student_id}/")
async def read_student(student_id: int):
    return {"lessons": 4}


uvicorn main:app --reload


Docker:

FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



docker build -t my_fastapi_app .


docker run -p 8000:8000 my_fastapi_app



######

'''
Yes, FastAPI supports generating an interface from a Swagger definition, using the fastapi.openapi.OpenAPI class.

You can define the OpenAPI schema for your API using the fastapi.openapi.OpenAPI class, and then use the fastapi.docs.get_openapi function to generate the OpenAPI documentation for your API.

Here is an example of how you can use the fastapi.openapi.OpenAPI class to generate an interface from a Swagger definition:
'''
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

openapi_schema = get_openapi(
    title="My Super Cool API",
    version="1.0.0",
    description="This is a very cool API",
    routes=app.routes,
)


'''
FastAPI has support for AsyncAPI, which is a specification for defining asynchronous APIs.

FastAPI uses the asyncio library to handle asynchronous code, so it's well suited for creating asynchronous APIs. It also provides a way to define asynchronous endpoints using the async def syntax.

FastAPI provides AsyncAPI support through the fastapi.asyncapi.AsyncAPI class and the fastapi.asyncapi.get_asyncapi function. These classes and functions are very similar to the fastapi.openapi.OpenAPI and fastapi.openapi.get_openapi functions and classes, that I mentioned earlier.

Here's an example of how you can use the fastapi.asyncapi.AsyncAPI class to define an asynchronous API with an AsyncAPI schema:
'''
from fastapi import FastAPI
from fastapi.asyncapi import AsyncAPI, Message

app = FastAPI()

async def consume_message(message: Message):
    print(message.payload)

asyncapi = AsyncAPI(
    title="My Super Cool Asynchronous API",
    version="1.0.0",
    description="This is a very cool Asynchronous API",
    routes=[
        {
            "route": "/messages/{item_id}",
            "operations": [
                {
                    "operationId": "consume_message",
                    "description": "Consume a message",
                    "message": {
                        "messageId": "message_id",
                        "payload": {
                            "type": "object",
                            "properties": {
                                "item_id": {"type": "integer"}
                            }
                        }
                    },
                    "consumes": consume_message
                }
            ]
        }
    ]
)


####
'''
You can use the @app.get decorator to define the endpoint and the path parameters, 
and then use the Operation class from fastapi.openapi to define the request body, 
responses and other information about the endpoint and use it to generate the OpenAPI schema, like this:
'''
from fastapi import FastAPI, Path
from fastapi.openapi.docs import get_operation

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

operation = get_operation(app, "/items/{item_id}")


## From a data contract in JSON to an endpoint in FastAPI

'''
Yes, you can go from a data contract in JSON to an endpoint in FastAPI.
There are several ways to do this, but here's a general approach you can take:

Define the data contract in JSON format. This can include the structure of the request and response bodies, as well as any path or query parameters.

Use a library or tool to convert the JSON data contract into Python classes. This can be done manually, but there are also several libraries that can help with this process such as pydantic, marshmallow, jsonschema and json2python.

Use the Python classes to define the endpoint in FastAPI. You can use the FastAPI class to define the endpoint, path parameters and query parameters, and use the Pydantic models to define the request and response bodies.

Implement the endpoint's logic, using the request and response classes.

Test the endpoint and make any necessary adjustments.

Pydantic can be used to generate the Python classes from a JSON schema, which is the format of the data contract in my example.

You can use the pydantic.create_model function to generate the Pydantic models from a JSON schema. Here's an example of how you can use this function to generate the Pydantic models from the JSON data contract:

'''




from fastapi import FastAPI
from pydantic import create_model

Item, ItemOut = create_model("Item", data_contract['paths']['/items/{item_id}']['get']['responses']['200']['content']['application/json']['schema'])

app = FastAPI()
@app.get("/items/{item_id}")
def read_item(item: Item):
    return ItemOut(**item.dict())


# fastAPI vs Sanic
'''
In general, if you're building an API that needs to handle a large number of requests, and you want to use the latest features of Python, you should consider using FastAPI.
On the other hand, if you're building an application that needs to handle a high number of concurrent clients and you want a lightweight and easy-to-use framework, you should consider using Sanic.
'''


'''
FastAPI supports GraphQL through the use of third-party libraries. There are several libraries available for adding GraphQL support to FastAPI, such as graphene, graphene-sqlalchemy, graphene-pydantic, fastapi-graphene and many more.

These libraries provide various features such as:

Defining a GraphQL schema
Defining resolvers for querying and mutating the data
Integrating with ORMs like SQLAlchemy
Automatic conversion of Pydantic models to GraphQL types
Here's an example of how you can use graphene and graphene-sqlalchemy to add GraphQL support to a FastAPI application:
'''
from fastapi import FastAPI
from graphene import ObjectType, String, Schema
from graphene_sqlalchemy import SQLAlchemyObjectType
from sqlalchemy.orm import Session

app = FastAPI()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class UserType(SQLAlchemyObjectType):
    class Meta:
        model = User

class Query(ObjectType):
    users = graphene.List(lambda: UserType)

    async def resolve_users(self, info):
        query = User.get_query(info)
        return await query.all()

schema = Schema(query=Query)

app.add_route("/graphql", GraphQLView.as_view(schema=schema))



