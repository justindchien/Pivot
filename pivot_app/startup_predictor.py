from flask import Flask, jsonify, request, render_template

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it
# patients = pd.read_csv("CB_final.csv", header=0)
# patients.columns=['ratingStatus','numberOfRatings','recommendToFriendRating','ceopctapprove']
# patients = patients.dropna()
#
# X = patients.drop('statusID',1)
# X_ordered = X.reindex_axis(sorted(X.columns), axis=1)
# Y = patients['statusID']

pkl = open("model_xgb.pkl", "rb")
#PREDICTOR = LogisticRegression().fit(X,Y)
PREDICTOR = pickle.load(pkl)#XGBClassifier().fit(X_ordered,Y)
pkl.close()

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Homepage
@app.route("/")
def index():
    """
    Homepage: serve our visualization page, index.html
    """
    return render_template("index.html")

# changes text to goodbye when you add '/goodbye' to the url
# any time you typed in a name you could format a page specifically
# for them from stored values on a server
@app.route('/map/')
def map():
	return render_template("map.html")


# # Get an example and return it's score from the predictor model
# Get an example and return it's score from the predictor model
@app.route("/score/", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = request.json
    print(data["example"])
    x = pd.DataFrame(data["example"],index=['i',])
    x_ordered = x.reindex_axis(sorted(x.columns), axis=1)
    score = PREDICTOR.predict_proba(x_ordered)
    # Put the result in a nice dict so we can send it as json

    results = {"oper":str(score[0,1]), "close":str(score[0,0]), "acq":str(score[0,2]), "ipo":str(score[0,3])}
    return jsonify(results)



#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
