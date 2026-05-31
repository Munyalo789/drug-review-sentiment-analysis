# Drug Review Sentiment Analysis - Updated Version

## Updated For
- Python 3.12
- TensorFlow 2.16
- Flask 3.x
- Keras 3.x

## Installation

Create virtual environment:

### Windows
python -m venv venv
venv\Scripts\activate

### Linux / Mac
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

## Run Application

python app.py

Open:
http://127.0.0.1:5000

## Important Notes

- Removed deprecated `predict_classes()`
- Updated TensorFlow / Keras imports
- Fixed tokenizer misuse
- Improved prediction handling
- Compatible with latest Python versions