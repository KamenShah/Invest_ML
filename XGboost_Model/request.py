import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'month':8,'day':26,'year':2019})
print(r.json())
#print(r.json())