# Aruspex-ML

## ZKML Performance Dashboard
This dashboard visualizes the performance of our ZKML system for autonomous weapons verification. It displays proof generation times and accuracy metrics for simulated missions, using Palantir's Blueprint toolkit.

To run the dashboard:
1. Navigate to the 'zkml-dashboard/' directory
2. Run 'npm install' to install dependencies
3. Run 'npm dev' to launch the dashboad (uses Vite)

## Threat Classifier
A TensorFlow neural network that takes signals, predicts threat probabilities, and outputs odds.

#### Usage
```
pip install tensorflow
```
```
python threat_classifier.py
```

#### Result
```
Signal 5.5 -> Threat probability: 0.xx
```

## Cost Predictor: Linear Regression Forecast

#### Usage
```
pip install tensorflow
```
```
python cost_predictor.py
```

#### Result
```
Month 6 cost: 587.69
```

#### Automation
GitHub Action tests 'threat_classifier.py' on push


#### License
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
