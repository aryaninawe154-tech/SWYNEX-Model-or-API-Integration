# SWYNEX-Model-or-API-Integration
AI-based prototype for classifying college computer laboratory issues.


This project is a small NLP-based prototype for classifying common college computer laboratory complaints.
The prototype uses TF-IDF for text feature extraction and Logistic Regression for issue classification.
Student Complaint
        ↓
Text Input
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression Model
        ↓
Predicted Issue Category
The model currently classifies complaints into:
- Network
- Hardware
- Software
- Login/Account
- Programming
Input:

"Wi-Fi is connected but internet is not working"

Output:

"Network"

### Example 2

Input:

"The keyboard is not responding"

Output:

"Hardware"

### Example 3

Input:

"VS Code is not opening"

Output:

"Software"

### Example 4

Input:

"I cannot login to the lab computer"

Output:

"Login/Account"

### Example 5

Input:

"My Java program is not compiling"

Output:

"Programming"

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Installation

Install the required dependency using:

pip install -r requirements.txt

## Running the Prototype

Run the following command:

python app.py

The program will display the student complaint and the predicted issue category.

## Sample Output

Student Complaint: Wi-Fi is connected but internet is not working

Predicted Category: Network

## Limitations

- The prototype uses a small sample dataset.
- The current model focuses on English complaints.
- The model is intended as a baseline prototype.
- More training data would be required for reliable real-world deployment.

## Future Improvements

- Expand the training dataset
- Add priority prediction
- Add a web interface
- Support multiple languages
- Compare different classification models
- Add automatic troubleshooting suggestions


## Actual Test Results
The prototype was tested locally using Python.

### Test 1
Input: "Wi-Fi is connected but internet is not working"
Output: Network

### Test 2
Input: "Keyboard is not working"
Output: Hardware

### Test 3
Input: "VS Code is not opening"
Output: Software

### Test 4
Input: "I cannot login to the lab computer"
Output: Login/Account

### Test 5
Input: "My Java program is not compiling"

Output: Programming
All five test inputs were successfully processed by the trained classification model.
