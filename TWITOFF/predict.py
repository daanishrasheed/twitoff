"""Prediction of Users based on Tweet embeddings."""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from .model import User
from .twitter import BASILICA

def predict_user