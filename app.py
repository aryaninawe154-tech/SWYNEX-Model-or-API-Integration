
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
    "Network", "Network", "Network",
    "Hardware", "Hardware", "Hardware",
    "Software", "Software", "Software",
    "Login/Account", "Login/Account", "Login/Account",
    "Programming", "Programming", "Programming", "Programming"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(complaints)

model = LogisticRegression()
model.fit(X, categories)


def get_solution(category):
    solutions = {
        "Network": "Check Wi-Fi and internet connectivity.",
        "Hardware": "Check device connections and restart the computer.",
        "Software": "Restart the application or reinstall it if needed.",
        "Login/Account": "Check your credentials or contact lab support.",
        "Programming": "Check syntax errors and read the error message."
    }
    return solutions.get(category, "Contact lab support.")


def handle_complaint(complaint):
    if not complaint or not complaint.strip():
        return "Error: Please enter a complaint."

    complaint_vector = vectorizer.transform([complaint])
    probabilities = model.predict_proba(complaint_vector)[0]
    best_index = probabilities.argmax()

    category = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < 0.30:
        return (
            "Complaint unclear. Please provide more details "
            "or contact lab support."
        )

    return (
        f"Complaint: {complaint}\n"
        f"Predicted Category: {category}\n"
        f"Confidence: {confidence:.2%}\n"
        f"Suggested Solution: {get_solution(category)}"
    )


print("--- TASK 3: INTELLIGENT ISSUE CLASSIFIER ---")

test_complaints = [
    "Wi-Fi is not working",
    "Keyboard is broken",
    "I cannot login to my account",
    "My Java program is not compiling",
    ""
]

for complaint in test_complaints:
    print("\nTesting:", repr(complaint))
    print(handle_complaint(complaint))


y_pred = model.predict(X)

print("\n--- MODEL EVALUATION ---")
print(
    "Training-data accuracy:",
    f"{accuracy_score(categories, y_pred) * 100:.2f}%"
)
print(classification_report(categories, y_pred, zero_division=0))
