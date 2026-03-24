from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Abstract base for all signal source loaders.

    All loaders return the same dict shape:

        {
            'x': np.ndarray,
            'y': np.ndarray,
            'metadata': {
                'has_ms_data': bool,   # guaranteed key
                'filename': str,       # guaranteed key
                # all other keys are loader-specific; access with .get()
            }
        }
    """

    @abstractmethod
    def load(self, c_folder_path: str) -> dict:
        """Load signal data from the given .C folder path."""
