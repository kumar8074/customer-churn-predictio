### In Order to run this project locally

1. Create a virtual Environment:
- conda create --prefix ./cmpenv python=3.10 (or)
- conda create -n cmpenv python=3.10

2. Install All the requirements
- pip install -r requirements.txt

3. start mlflow
- mlflow ui (in terminal) (This should create a mlruns folder)

4. Run the training pipeline 
- python src/pipeline/train_pipeline.py
- This will create artifacts folder(which contains `model.pkl`,`preprocessor.pkl`) a logs folder(which contains the logs of trainig pipeline) and a mlartifacts folder.
- This will also give a URL in terminal which will redirect to mlflow UI for experiment tracking.
- Verify all these before proceeding.

5. Once Everything looks good run the flask app
- python app.py (it will start the flask app at 5001 port)

- In the flask upload the test data ('DATA/test_data.csv') and click on the predict button

- You can even use other data but make sure that the data is of the same shape as `test.csv`

- The flask app could have been implemented in such a way that we could have created seperate text feilds for each feature but i took the liberty to implement it like this for ease of use.

- Incase you want to test components individually Run:
   - python test_components.py
   For testing pipelines individually Run:
   - python test_train_pipeline.py
   - python test_predict_pipeline.py

- NOTE That the detailed explanation of everything is present in jupyter notebook at `Notebooks/customer_churn_prediction.ipynb`


### Important Note
Dear Evaluator testing ML projects could be tedeous you need to create environments, download libraries, run a bunch of commands etc.,etc., Hence i wanted to make your job easier by containarizing application and then pulishing the image on `DockerHub` so that you can easily just pull the image and run it to test if application is working fine or not. However i had a very little time as my end semester exams are going on so i was just working on this project whenever i got free time so unfortunately i couldn't complete it. So yeah it's still work in progress...so please ignore the DockerFiles 