# SWYNEX-Model-or-API-Integration

## Campus Lab Issue Classifier 🤖

An AI-based prototype that classifies common college computer laboratory complaints and suggests possible troubleshooting solutions.

## How It Works

Student Complaint → TF-IDF Feature Extraction → Logistic Regression → Predicted Category → Suggested Solution

## Supported Categories

- Network
- Hardware
- Software
- Login/Account
- Programming

## Features

- NLP-based complaint classification
- TF-IDF text feature extraction
- Logistic Regression model
- Prediction confidence score
- Automatic troubleshooting suggestions
- Empty-input validation
- Low-confidence handling
- Model evaluation report

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Installation

Install the required dependency:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

## Example

**Input:** My Java program is not compiling

**Predicted Category:** Programming

**Suggested Solution:** Check syntax errors and read the error message.

## Error Handling

- Empty complaints return an error message asking the user to enter a complaint.
- Low-confidence predictions ask the user to provide more details or contact lab support.

## Evaluation Results

The model achieved **100% accuracy on its 16 training examples**.

This is training-data accuracy, not independent test accuracy. Further testing on unseen complaints is required to evaluate real-world performance.

The prototype also generates a classification report containing precision, recall, F1-score, and support for each category.

## Limitations

- The training dataset is small.
- The model currently focuses on English complaints.
- Confidence scores are estimates, not guarantees of correctness.
- The prototype requires evaluation on more unseen examples before real-world use.

## Future Improvements

- Expand the training dataset.
- Evaluate on unseen complaints.
- Add priority prediction.
- Develop a web interface.
- Support multiple languages.
- Compare different classification models.

## Author

Arya Ninawe
