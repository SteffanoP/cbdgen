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

    def __init__(self, options: dict):
        self._samples = options['samples']
        self._attributes = options['attributes']
        self._classes = options['classes']
        self._optional_option = options['maker'][1]

    def generate(self, type_gen: int) -> DataFrame:
        return getattr(self, self.types_generator.get(type_gen))()

    def _blobs(self) -> DataFrame:
        centers = self._optional_option
        return maker.blobs(self._samples, centers, self._attributes)

    def _moons(self) -> DataFrame:
        noise = self._optional_option
        return maker.moons(self._samples, noise)

    def _circles(self) -> DataFrame:
        noise = self._optional_option
        return maker.circles(self._samples, noise)

    def _classf(self) -> DataFrame:
        return maker.classification(self._samples, self._attributes, 
                                    self._classes)

    def _mlabel_classf(self) -> DataFrame:
        labels = self._optional_option
        return maker.multilabel_classification(self._samples, self._attributes, 
                                               self._classes, labels)
