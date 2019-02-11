
class Workflow:
    def __init__(self, stages, storage):
        self.stages = stages
        self.storage = storage

    @classmethod
    def from_JSON(cls, json_object):
        stages = json_object['stages']
        return cls(stages)

    def run(self, scope={}):
        scope = dict(scope)
        for stage in self.stages:
            stage.run(scope, self.storage)
