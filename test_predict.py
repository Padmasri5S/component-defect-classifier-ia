import requests
url = "http://127.0.0.1:8000/predict"
file_path = r"C:\\Users\\ps18286\\OneDrive - Surbana Jurong Private Limited\\Desktop\\ImageAnnot\\images\\Circular Connection_Drummy\\P1127_D169.jpg"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})
print(response.json()["prediction"])
print(response.status_code)
print(response.text)
try:
    print(response.json())
except Exception as e:
    print("Failed to decode JSON:", e)


print(response.json())
print(response.status_code)
print(response.text)
