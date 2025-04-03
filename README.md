# Aruspex-ML: Predictions with TensorFlow

## Threat Classifier
A TF neural network that takes signals, predicts threat probabilities, and outputs odds.

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
