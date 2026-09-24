from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
complaints = [
    "Wi-Fi is not working",
    "Internet is disconnected",
    "Network keeps disconnecting",
    "Keyboard is not working",
    "Mouse is damaged",
    "Computer monitor is not displaying",
    "VS Code is not opening",
    "Browser keeps crashing",
    "Software is not installed",
    "I cannot login to the lab computer",
    "My lab password is not working",
    "I cannot access my account",
    "My PHP code is showing an error",
    "Java program is not compiling",
    "Python program is giving an error",
    "HTML page is not displaying correctly"
]

categories = [
    "Network",
    "Network",
    "Network",
    "Hardware",
    "Hardware",
    "Hardware",
    "Software",
    "Software",
    "Software",
    "Login/Account",
    "Login/Account",
    "Login/Account",
    "Programming",
    "Programming",
    "Programming",
    "Programming"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(complaints)

model = LogisticRegression()
model.fit(X, categories)


def classify_issue(complaint):
    complaint_vector = vectorizer.transform([complaint])
    prediction = model.predict(complaint_vector)[0]

    return prediction

user_complaint = "Wi-Fi is connected but internet is not working"

result = classify_issue(user_complaint)

print("Student Complaint:", user_complaint)
print("Predicted Category:", result)


