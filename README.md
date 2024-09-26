# Movie Recommendation System

## Description
This Flask-based Movie Recommendation System utilizes collaborative filtering and natural language processing to provide personalized movie recommendations based on user preferences. It leverages GloVe embeddings for efficient similarity calculations.

## Features
- User-friendly web interface for entering User ID
- Movie recommendations based on user history
- Displays cosine similarity of recommendations
- Built with Flask, Pandas, and Scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a virtual environment (if you haven't already):
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Place your `metadata.csv` and `interaction.csv` files in the `data` directory.

## Usage
Run the application:
```
flask run
```
Visit `http://127.0.0.1:5000` in your web browser.

### Example
![Screenshot of the application](images/screenshot.png)

## GIF Demonstration
![Demo GIF](images/demo.gif)


## Acknowledgments
- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [GloVe](https://nlp.stanford.edu/projects/glove/)
