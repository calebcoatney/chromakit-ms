from PySide6.QtCore import QObject, QTimer, Signal, Slot, Qt

class MainThreadDispatcher(QObject):
    """Utility class to run functions on the main thread."""
    
    dispatch_signal = Signal(object)
    
    def __init__(self):
        super().__init__()
        # Use Qt.QueuedConnection to ensure cross-thread safety
        self.dispatch_signal.connect(self._dispatch, Qt.QueuedConnection)
    
    @Slot(object)
    def _dispatch(self, func):
        """Execute the function on the main thread."""
        try:
            func()
        except Exception as e:
            print(f"Error in main thread dispatch: {e}")
    
    def run_on_main_thread(self, func):
        """Schedule a function to run on the main thread."""
        # Just emit the signal - it will be queued and executed on main thread
        self.dispatch_signal.emit(func)

# Create a global instance
main_thread_dispatcher = MainThreadDispatcher()