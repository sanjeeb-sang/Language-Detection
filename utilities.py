from typing import List


def does_list_have_value(list_to_check: List) -> bool:
    """
    Method that checks if the list has more than zero values.
    :param list_to_check: the list which is to be checked.
    :return: True if the list_to_check is not none and has atleast one value.
    """
    return list_to_check is not None and len(list_to_check) > 0


def is_list_empty(list_to_check: List) -> bool:
    """
    Method to check if the passed list if empty or not.
    :param list_to_check: the list to check.
    :return: True if the passed list has no value, false if the list has atleast one value.
    """
    return list_to_check is None or len(list_to_check) > 0


def is_valid_index(index_to_check: int, length_of_list_or_string: int) -> bool:
    """
    Method to check if the passed index is valid for a list whose length is given.
    :param index_to_check: the index of a list to check.
    :param length_of_list_or_string: the length of the associated list.
    :return: true if the passed index is valid, false otherwise.
    """
    return index_to_check < length_of_list_or_string


def is_string_valid(string_to_check: str) -> bool:
    """
    Method to check if the passed string value is valid (or is not empty).
    :param string_to_check: the string value which needs to be checked if it's empty or not.
    :return: true if the passed string is not empty, false if the passed string is empty.
    """
    return string_to_check is not None and len(string_to_check) > 0





