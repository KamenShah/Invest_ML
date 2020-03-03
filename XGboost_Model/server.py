# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
from dataRetrieval import *


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route('/api',methods=['POST'])
def predict():
	data = request.get_json(force=True)
	predictDate  = datetime.date(data["year"], data["month"], data["day"])
	output = predictEarnings(predictDate, model)
	return output


if __name__ == '__main__':
    app.run(port=5000, debug=True)


