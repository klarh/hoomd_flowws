import contextlib

import hoomd
import flowws

def intfloat(x):
    """int . float (for convenience in command-line specification)"""
    return int(float(x))

class RestoreStateContext:
    def __init__(self, scope, storage, restore):
        self.scope = scope
        self.storage = storage
        self.restore = restore

    def try_restoring(self):
        restore_filename = self.scope.get('restore_filename', 'backup.tar')

        with self.storage.open(restore_filename, 'rb', on_filesystem=True) as f:
            system = hoomd.init.read_getar(f.name)
        return system

    def try_saving(self):
        if self.scope is not None:
            restore_filename = self.scope.get('restore_filename', 'backup.tar')
            with self.storage.open(restore_filename, 'wb', on_filesystem=True) as f:
                hoomd.dump.getar.immediate(f.name, [], dynamic=['all'])

    def __enter__(self, *args, **kwargs):
        if self.restore:
            self.scope['system'] = self.try_restoring()
        return self

    def __exit__(self, *args, **kwargs):
        self.try_saving()

class HoomdContext(contextlib.ExitStack):
    def __init__(self, scope, storage, restore=True):
        super(HoomdContext, self).__init__()

        self.scope = scope
        self.storage = storage
        self.cached_snapshot_ = None

        self.context_args = self.scope.get('hoomd_context_args', '')

        self.hoomd_context = self.enter_context(
            hoomd.context.initialize(self.context_args))
        self.restore_context = self.enter_context(
            RestoreStateContext(scope, storage, restore))

    def check_timesteps(self):
        """Returns True and cancels saving when exiting this context if the
        stage has already been completed (i.e. it can be skipped)
        """
        passed = hoomd.get_step() >= self.scope.get('cumulative_steps', 0)

        if passed:
            self.cancel_saving()

        return passed

    def try_restoring(self):
        return self.restore_context.try_restoring()

    def try_saving(self):
        return self.restore_context.try_saving()

    def cancel_saving(self):
        self.restore_context.scope = None

    @property
    def snapshot(self):
        if self.cached_snapshot_ is None:
            self.cached_snapshot_ = self.scope['system'].take_snapshot()
        return self.cached_snapshot_

class WorkflowError(RuntimeError):
    def __init__(self, msg):
        self.message = msg
