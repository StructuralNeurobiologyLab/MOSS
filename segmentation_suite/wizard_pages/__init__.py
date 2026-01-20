"""
Wizard page widgets for the training workflow.
"""

from .setup_page import SetupPage
from .training_page import TrainingPage
from .reslice_page import ReslicePage
from .prediction_page import PredictionPage
from .voting_page import VotingPage
from .finish_page import FinishPage

__all__ = [
    'SetupPage',
    'TrainingPage',
    'ReslicePage',
    'PredictionPage',
    'VotingPage',
    'FinishPage',
]
