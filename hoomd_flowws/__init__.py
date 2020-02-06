from flowws import try_to_import

from .version import __version__

Damasceno2017Interaction = try_to_import('.Damasceno2017Interaction', 'Damasceno2017Interaction', __name__)
DEMInteraction = try_to_import('.DEMInteraction', 'DEMInteraction', __name__)
Init = try_to_import('.Init', 'Init', __name__)
Interaction = try_to_import('.Interaction', 'Interaction', __name__)
Run = try_to_import('.Run', 'Run', __name__)
RunHPMC = try_to_import('.RunHPMC', 'RunHPMC', __name__)
ShapeDefinition = try_to_import('.ShapeDefinition', 'ShapeDefinition', __name__)
