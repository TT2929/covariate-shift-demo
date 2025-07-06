# covariate-shift-demo
This project presents an interactive web demonstration of the covariate shift problem in machine learning and how it can be mitigated using weighted Support Vector Machines (SVM). Users can manipulate the test data distribution in real-time and visually observe the impact of KDE or KMM-based weighting on the model's decision boundary.


##Motivation
"Covariate shift", where the distributions of training and test data differ, is a common challenge in real-world machine learning models. This project was developed to gain an intuition for this problem and visualize the effect of weighting techniques that solve it.


##Feature
- You can change the test distribution by moving the slider.
- You can compare the two weighting methods, KDE and KMM.
- Outputs decision boundaries and accuracy in real time.

## Setup
Programs are implemented in Python.
The following packages can be installed via `pip` or `conda`:

- `numpy`
- `streamlit`
- `matplotlib`
- `math`
- `scikit-learn`
- `logging`


## Usage
Run the following.
- `streamlit run main.py`
