from predict_risk import predict_risk

patient_a = dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2,
                 thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6)
patient_b = dict(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0, restecg=2,
                 thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3)

print("A:", predict_risk(patient_a))
print("B:", predict_risk(patient_b))
