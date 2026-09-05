from dataclasses import dataclass

from training.checkpoint_utils import load_k400_backbone_strict, replace_classifier


@dataclass
class TrainingAdapter:
    model: object
    classifier_path: str

    def initialize_from_k400(self, checkpoint, metadata_path):
        report = load_k400_backbone_strict(self.model, checkpoint, self.classifier_path, metadata_path)
        replace_classifier(self.model, self.classifier_path, 174)
        return report

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def __call__(self, inputs):
        return self.model(inputs)
