from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class FlowerForm(FlaskForm):
    COUNTRY_CODE = StringField("COUNTRY_CODE")
    MCC_CODE = StringField("MCC_CODE")
    US_TRAN_AMT = StringField("US_TRAN_AMT")
    submit = SubmitField("Predict")
