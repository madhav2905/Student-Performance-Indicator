import pandas as pd 
from flask import Flask,request,render_template 
from src.pipeline.predict_pipeline import PredictPipeline 

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict(): 
    if request.method=='POST': 
        data={
            "gender": request.form.get("gender"),
            "race/ethnicity": request.form.get("race_ethnicity"),
            "parental level of education": request.form.get("parental_education"),
            "lunch": request.form.get("lunch"),
            "test preparation course": request.form.get("test_prep"),
            "reading score": float(request.form.get("reading_score")),
            "writing score": float(request.form.get("writing_score"))
        }

        input_df=pd.DataFrame([data])

        pipeline=PredictPipeline()
        prediction=pipeline.predict(input_df)[0]

        return render_template('index.html',prediction=round(prediction,2) if prediction is not None else None)
    
# if __name__=="__main__":
#     app.run(debug=True)