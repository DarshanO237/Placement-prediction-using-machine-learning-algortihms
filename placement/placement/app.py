import numpy as np
import model
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    cgpa= request.args.get('cgpa')
    projects = request.args.get('projects')
    workshops = request.args.get('workshops')
    mini_projects = request.args.get('mini_projects')
    skills = request.args.get('skills')
    communication_skills = request.args.get('communication_skills')
    internship = request.args.get('internship')
    hackathon = request.args.get('hackathon')
    tw_percentage = request.args.get('tw_percentage')
    te_percentage = request.args.get('te_percentage')
    arr = np.array([cgpa,projects,workshops,mini_projects,skills,communication_skills,internship,hackathon,tw_percentage,te_percentage])
    brr = np.asarray(arr,dtype=float)
    output = model.predict([brr])
    if(output=='Placed'):
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
    return render_template('out.html', output=out)
     

if __name__ == "__main__":
    app.run(debug=True)
