import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text

import numpy as np
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class CustomIntentClassifier(Component):
    """A custom component to classify intent from a transliterated_text_features from the message."""

    requires = [
        "latin_text_features"
    ]

    provides = [
        "intent"
    ]
    # None means supports all languages.
    # Don't include ```language: en``` in your config.yaml
    language_list = None

    @classmethod
    def required_packages(cls):
        return ["sklearn"]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 clf: "CustomIntentClassifier" = None
                 ) -> None:
        super(CustomIntentClassifier, self).__init__(component_config)
        self.clf = clf

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Train intent classifier."""
        # intent names are hacked two content two types of information
        X = []
        y = []
        for example in training_data.intent_examples:
            latin_text_features = example.get("latin_text_features")
            intent = example.get("intent")
            intent = intent[3:] if intent.startswith("hi_") else intent
            X.append(latin_text_features)
            y.append(intent)

        X = np.array(X)
        y = np.array(y)

        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression(penalty="l1")
        from sklearn.svm import SVC
        clf = SVC(kernel="linear", probability=True, C=10)

        clf.fit(X, y)

        logging.info("classification score: {}".format(clf.score(X, y)))

        self.clf = clf

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""
        classifier_file_name = file_name+"_classifier.pkl"
        if self.clf:
            utils.json_pickle(os.path.join(
                model_dir, classifier_file_name), self.clf)
        return {"classifier": classifier_file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["CustomIntentClassifier"] = None,
        **kwargs: Any
    ) -> "CustomIntentClassifier":
        from sklearn.svm import SVC
        # from sklearn.linear_model import LogisticRegression

        classifier_file = os.path.join(model_dir, meta.get("classifier"))

        if os.path.exists(classifier_file):
            classifier = utils.json_unpickle(classifier_file)
            return cls(meta, classifier)

        return cls(meta)
    
    def process(self, message: Message, **kwargs: Any) -> None:
        """Return intent of the latin text for a message."""
        latin_text_features = message.get("latin_text_features")
        probs = self.clf.predict_proba([latin_text_features])[0]
        index = np.argmax(probs)
        confidence = probs[index]
        intent = self.clf.classes_[index]
        intent = {"name": intent, "confidence": confidence}

        message.set("intent", intent, add_to_output=True)

