python_flatten_json.py



# from here https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
def flatten_data(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


# From https://link.medium.com/YwiksTYew2
def flattenjson(b):
    delim = '.'
    v={}
    for i in collections.OrderedDict(b).keys():
        fname=i.replace("fields","")
        if isinstance(b[i],dict):
            get=flattenjson(b[i])
            for j in get.keys():
                if fname+delim==delim:
                    v[j]=get[j]
                else:
                    v[fname+delim+j]=get[j]
        elif isinstance(b[i],list):
            for l in range(len(b[i])):

                if isinstance(b[i][l],dict):
                    get1=flattenjson(b[i][l])

                    for t in get1.keys():
                        v[fname+delim+t]=get1[t]
        else:
            v[fname]=b[i]
    return collections.OrderedDict(v)