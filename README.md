# Movie Recommendation System with Flask, GloVe using Content Based Filtering


## Example/GIF Demonstration
![Demo GIF](images/Demo.gif)

## Description
This project is a **Movie Recommendation System** developed using **Flask** and **GloVe** embeddings, employing **Content-Based Filtering** techniques. The application recommends movies based on user preferences by analyzing the content of the movies.

## Features
- Utilizes **GloVe** for embedding movie descriptions.
- Implements a **Content-Based Filtering** algorithm to generate recommendations.
- User-friendly interface built with **Bootstrap** for responsiveness and elegance.

## Technologies Used
- **Flask**: A lightweight WSGI web application framework.
- **GloVe**: Global Vectors for Word Representation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **scikit-learn**: For computing cosine similarity.

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