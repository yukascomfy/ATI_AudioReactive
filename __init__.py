from .audio_reactive_path_animator import AudioReactivePathAnimator

NODE_CLASS_MAPPINGS = {
    "AudioReactivePathAnimator": AudioReactivePathAnimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioReactivePathAnimator": "Audio Reactive Path Animator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

WEB_DIRECTORY = "./js"
