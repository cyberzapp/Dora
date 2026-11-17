"""Conservative disk budget for cached RGB windows and final assembly."""
import shutil


def require_render_space(directory, frames, width, height):
    if any(type(value) is not int or value <= 0 for value in (frames, width, height)):
        raise ValueError('Positive integer frame geometry required')
    # Cached RGB windows plus assembled RGB, with room for checkpoints, encoding
    # and filesystem operation. Free space can still change during a long job.
    needed = frames * width * height * 3 * 2 + 2 * 1024**3
    available = shutil.disk_usage(directory).free
    if available < needed:
        raise OSError(f'Not enough free disk space for this render. Need approximately '
                      f'{needed / 1024**3:.1f} GB available; {available / 1024**3:.1f} GB is free. '
                      'Free space and resume this job; saved progress is retained.')
    return needed
