# platform_loader.py
import os
import platform

_loaded = False


def get_topic_module():
    """Load this product's Linux x86-64 Topic module."""
    global _loaded
    if not _loaded:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system != 'linux' or machine not in ('x86_64', 'amd64'):
            raise RuntimeError(f"Unsupported platform for product-local Topic module: {system}/{machine}")
        topic_path = os.path.join(base_dir, 'topic.so')
        if not os.path.isfile(topic_path):
            raise RuntimeError(f"Product Topic module not found: {topic_path}")

        _loaded = True

    import topic
    return topic
