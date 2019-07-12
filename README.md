# rasa_custom
Custom components for NLU to support multiple languages and multiple text scripts with Rasa bot.

Download [rasa x](https://rasa.com/docs/rasa-x/installation-and-setup/), then replace contents of the `rasa` folder from `site-packages` with contents of this repo/folder.

## Contributions:
1. [Custom pipeline](https://github.com/psds01/rasa_custom/blob/d54a827df0ee9535814693f51571b2fb73432dd6/nlu/registry.py#L135) 
2. Custom components:
    1. [custom extractors](https://github.com/psds01/rasa_custom/blob/d54a827df0ee9535814693f51571b2fb73432dd6/nlu/extractors/custom_extractors.py)
    2. [custom featurizer](https://github.com/psds01/rasa_custom/blob/d54a827df0ee9535814693f51571b2fb73432dd6/nlu/featurizers/custom_featurizer.py)
    3. [custom classifier](https://github.com/psds01/rasa_custom/blob/d54a827df0ee9535814693f51571b2fb73432dd6/nlu/classifiers/custom_intent_classifier.py)
