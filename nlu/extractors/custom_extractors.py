import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple

import numpy as np
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SCHEMES, SchemeMap, transliterate
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData


class ScriptExtractor(Component):
    """Does not require any kind of featurization."""
    provides = ["script"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(ScriptExtractor, self).__init__(component_config)

    def get_script(self, text: Text) -> Text:
        """
        Very stOOpid logic to extract devanagari.
        Detect script from the text message.
        If ascii characters than 50%, then latin else devanagari script.
        """
        # assume latin to be the default script
        script = "latin"

        count = 0
        for ch in text:
            if ord(ch) < 128:
                count += 1
        len_text = len(text) or 1  # avoid zero division

        if count/len_text < 0.5:
            script = "devanagari"

        return script

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Not a trainable component."""
        for example in training_data.intent_examples:
            script = self.get_script(example.text)
            example.set("script", script, add_to_output=True)
        return None

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        return None

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional["ScriptExtractor"] = None,
             **kwargs: Any) -> "ScriptExtractor":
        return cls(meta)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return script of the text for a message."""
        script = self.get_script(message.text)
        message.set("script", script, add_to_output=True)


class LatinTextExtractor(Component):
    """Does not require any kind of featurization."""
    requires = [
        "script"
    ]
    provides = [
        "latin_text"
    ]

    @classmethod
    def required_packages(cls):
        return ["indic_transliteration"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(LatinTextExtractor, self).__init__(component_config)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Not a trainable component."""
        for example in training_data.intent_examples:
            latin_text = self.get_latin(example)
            example.set("latin_text", latin_text, add_to_output=True)
        return None

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Nothing to persist here."""
        return None

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional["ScriptExtractor"] = None,
             **kwargs: Any) -> "ScriptExtractor":

        return cls(meta)

    def get_latin(self, message: Message) -> Text:
        text = message.text
        script = message.get("script")
        latin_text = text
        if script == "devanagari":
            latin_text = transliterate(
                text, sanscript.DEVANAGARI, sanscript.OPTITRANS).lower()
        return latin_text

    def process(self, message: Message, **kwargs: Any) -> None:
        """Convert text into latin from devanagari."""
        latin_text = self.get_latin(message)
        message.set("latin_text", latin_text, add_to_output=True)


class LanguageExtractor(Component):
    """Requires features."""
    requires = [
        "latin_text_features",
        # "script" -> bad idea!
    ]
    provides = [
        "language"
    ]

    @classmethod
    def required_packages(cls):
        return ["sklearn"]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 clf: "LanguageExtractor" = None
                 ) -> None:
        super(LanguageExtractor, self).__init__(component_config)
        self.clf = clf

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Train language classifier."""
        X = []
        y = []
        for example in training_data.intent_examples:
            latin_text_features = example.get("latin_text_features")
            # Well this is hacked!!
            intent = example.get("intent")
            language = "hi" if intent.startswith("hi_") else "en"
            example.set("language", {"name": language, "confidence": 1.0})
            X.append(latin_text_features)
            y.append(language)

        X = np.array(X)
        y = np.array(y)

        # from sklearn.svm import SVC
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
        cached_component: Optional["LanguageExtractor"] = None,
        **kwargs: Any
    ) -> "LanguageExtractor":
        from sklearn.svm import SVC
        # from sklearn.linear_model import LogisticRegression

        classifier_file = os.path.join(model_dir, meta.get("classifier"))

        if os.path.exists(classifier_file):
            classifier = utils.json_unpickle(classifier_file)
            return cls(meta, classifier)

        return cls(meta)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return language of the latin text for a message."""
        latin_text_features = message.get("latin_text_features")
        probs = self.clf.predict_proba([latin_text_features])[0]
        index = np.argmax(probs)
        confidence = probs[index]
        language = self.clf.classes_[index]
        language = {"name": language, "confidence": confidence}

        message.set("language", language, add_to_output=True)
