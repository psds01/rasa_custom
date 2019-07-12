import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Text

import numpy as np
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


class CustomFeaturizer(Featurizer):
    requires = [
        "latin_text"
    ]

    provides = [
        "latin_text_features"
    ]
    defaults = {
        "min_count": 1
    }

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 grams_mapping: Dict[Text, Any] = None) -> None:

        super(CustomFeaturizer, self).__init__(component_config)

        if grams_mapping is None:
            grams_mapping = {}
        self.grams_mapping = grams_mapping

    def _make_grams(self, text):
        """Makes bigrams and 2-skipgrams from the text."""
        grams = defaultdict(int)
        for txt in text.split():
            for i in range(len(txt)-1):
                grams[txt[i:i+2]] += 1
            for i in range(len(txt)-2):
                grams[txt[i]+"_"+txt[i+2]] += 1
        return grams

    def featurize(self, text):
        arr = np.zeros(len(self.grams_mapping))
        grams = self._make_grams(text)
        for key, value in grams.items():
            if key in self.grams_mapping:
                arr[self.grams_mapping[key]] += value
        return list(arr)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Make a bigram and skipgram lookup table from all the *latin* text from the training data."""
        grams_counter = defaultdict(int)
        for example in training_data.intent_examples:
            latin_text = example.get("latin_text")
            grams = self._make_grams(latin_text)
            for key, value in grams.items():
                grams_counter[key] += value
        grams_mapping = {}
        index = 0
        for key, value in grams_counter.items():
            if value >= self.component_config['min_count']:
                grams_mapping[key] = index
                index += 1
        self.grams_mapping = grams_mapping
        for example in training_data.intent_examples:
            latin_text = example.get("latin_text")
            latin_text_features = self.featurize(latin_text)
            example.set("latin_text_features",
                        latin_text_features, add_to_output=False)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""
        mapping_file_name = file_name+"_mapping.pkl"
        if self.grams_mapping:
            utils.json_pickle(os.path.join(
                model_dir, mapping_file_name), self.grams_mapping)
        return {"grams_mapping": mapping_file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Optional[Text] = None,
            model_metadata: Optional[Metadata] = None,
            cached_component: Optional["CustomFeaturizer"] = None,
            **kwargs: Any) -> "CustomFeaturizer":

        mapping_file_name = os.path.join(model_dir, meta.get("grams_mapping"))
        if os.path.exists(mapping_file_name):
            grams_mapping = utils.json_unpickle(mapping_file_name)
            return cls(meta, grams_mapping)
        return cls(meta)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return features of the latin text for a message."""
        latin_text = message.get("latin_text")
        latin_text_features = self.featurize(latin_text)
        # Does `add_to_output` play a role in predicting next action?
        message.set("latin_text_features",
                    latin_text_features, add_to_output=False)
