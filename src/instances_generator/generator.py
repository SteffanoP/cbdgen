from pandas import DataFrame
import instances_generator.maker as maker

class InstancesGenerator:
    types_generator = {
        1: '_blobs',
        2: '_moons',
        3: '_circles',
        4: '_classf',
        5: '_mlabel_classf'
    }

    def __init__(self, options: dict) -> DataFrame:
        self.samples = options['samples']
        self.attributes = options['attributes']
        self.classes = options['classes']
        self.optional_option = options['maker'][1]

    def generate(self, type_gen: int) -> DataFrame:
        return getattr(self, self.types_generator.get(type_gen))()

    def _blobs(self) -> DataFrame:
        centers = self.optional_option
        return maker.blobs(self.samples, centers, self.attributes)

    def _moons(self) -> DataFrame:
        noise = self.optional_option
        return maker.moons(self.samples, noise)

    def _circles(self) -> DataFrame:
        noise = self.optional_option
        return maker.circles(self.samples, noise)

    def _classf(self) -> DataFrame:
        return maker.classification(self.samples, self.attributes, 
                                    self.classes)

    def _mlabel_classf(self) -> DataFrame:
        labels = self.optional_option
        return maker.multilabel_classification(self.samples, self.attributes, 
                                               self.classes, labels)
