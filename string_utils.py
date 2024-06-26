import copy


def remove_char_from_string(string, char):

    return string.replace(char, '')

def remove_list_of_chars_from_string(string, chars):

    for char in chars:
        string = remove_char_from_string(string=string, char=char)

    return string

def remove_substring_from_end_of_string(string, substring):

    if substring and string.endswith(substring):
        return string[:-len(substring)]
    else:
        raise ValueError(
            "{} does not end with {}".format(string, substring)
        )


def remove_substring_from_start_of_string(string, substring):

    if substring and string.startswith(substring):
        return string[len(substring):]
    else:
        raise ValueError(
            "{} does not start with {}".format(string, substring)
        )


def remove_substrings_from_start_and_end_of_string(string, substrings):

    # NOTE: substrings must be a list

    if len(substrings) == 2:

        return remove_substring_from_end_of_string(
            string=remove_substring_from_start_of_string(
                string=string,
                substring=substrings[0]
            ),
            substring=substrings[1]
        )

    else:
        raise ValueError("...")

    return string_temp



if __name__ == "__main__":

    pass
