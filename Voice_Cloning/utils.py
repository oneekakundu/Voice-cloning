from pathlib import Path


def ensure_directory(directory):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : str or Path
        Directory path.

    Returns
    -------
    Path
        Created or existing directory path.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def validate_text(text):
    """
    Validate text before sending it to XTTS.

    Parameters
    ----------
    text : str
        Text that will be converted into speech.

    Raises
    ------
    TypeError
        If the input is not a string.

    ValueError
        If the text is empty.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Text must be a string."
        )

    if not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    return text.strip()


def validate_user_id(user_id):
    """
    Validate a user ID.

    Parameters
    ----------
    user_id : str
        User identifier.

    Raises
    ------
    TypeError
        If the user ID is not a string.

    ValueError
        If the user ID is empty.
    """

    if not isinstance(user_id, str):
        raise TypeError(
            "User ID must be a string."
        )

    if not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    return user_id.strip()