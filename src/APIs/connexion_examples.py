'''
	From Zalando: 
	https://jobs.zalando.com/en/tech/blog/crafting-effective-microservices-in-python/

	pip3 install connexion --upgrade


	Useful links:
	https://opensource.zalando.com/restful-api-guidelines/

	https://github.com/zalando/connexion

	https://github.com/hjacobs/connexion-example


	From OpenAPI into GraphQL:
	https://github.com/praveenweb/openapi-swagger-remote-schema

	From PostgreSQL to API
	PostgREST is a standalone web server that turns your PostgreSQL database directly into a RESTful API
	http://postgrest.org/en/v7.0.0/index.html


	OpenAPI does not fully recognise/support JSON schema

'''

# run this command in the same directory where you saved the previous files.
$ pip install connexion # installs connexion, run only once.
$ connexion run my_api.yaml -v


# it cannot be...
import connexion

app = connexion.App(__name__, specification_dir='swagger/')
app.add_api('my_api.yaml')
app.run(port=8080)


# To use Tornado
import connexion

app = connexion.App(__name__, specification_dir='swagger/')
app.run(server='tornado', port=8080)