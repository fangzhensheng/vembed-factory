from typing import Any


def looks_like_image_path(content: Any) -> bool:
    """Check if content looks like an image file path.

    Args:
        content: The content to check.

    Returns:
        True if the content is a string that looks like an image file path, False otherwise.
    """
    if not content or not isinstance(content, str):
        return False

    content_lower = content.lower().strip()
    if not content_lower:
        return False

    # Check for image extensions
    if any(
        content_lower.endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")
    ):
        return True

    # Check for path separators (indicates a file path, not text content)
    return "/" in content_lower or "\\" in content_lower
