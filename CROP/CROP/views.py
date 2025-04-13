from django.shortcuts import render
import pickle
import os
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, "Random Forest Classification of Crop Prediction.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def home(request):
    prediction = None
    error = None

    if request.method == "POST":
        try:
            temp = float(request.POST.get('temp', '').strip())
            hue = float(request.POST.get('hue', '').strip())
            ph = float(request.POST.get('ph', '').strip())
            wa = float(request.POST.get('wa', '').strip())

            two_d_data = [[temp, hue, ph, wa]]
            prediction = model.predict(two_d_data)[0]

        except ValueError:
            error = "Please fill all the fields with valid numeric values."

    return render(request, "index.html", {"Prediction": prediction, "Error": error})
