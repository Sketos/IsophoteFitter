import os
import pickle

# NOTE:
import string_utils as string_utils


# # NOTE:
# def sanitize(filename):
#
#     if filename.endswith(".pickle"):
#         filename_sanitized = string_utils.remove_substring_from_end_of_string(
#             string=filename, substring=".pickle"
#         )
#     else:
#         filename_sanitized = filename
#     return "{}/{}.pickle".format(
#         directory, filename_sanitized
#     )


# NOTE:
def save_obj(directory, filename, obj, overwrite=False):

    def save(filename, obj):
        with open(filename, "wb+") as f:
            pickle.dump(obj, f)

    if filename.endswith(".pickle"):
        filename_sanitized = string_utils.remove_substring_from_end_of_string(
            string=filename, substring=".pickle"
        )
    else:
        filename_sanitized = filename

    filename = "{}/{}.pickle".format(
        directory, filename_sanitized
    )

    if os.path.isfile(filename):
        if overwrite:
            save(filename=filename, obj=obj)
            #raise ValueError("This has not been implemented yet")
        else:
            print(
                "The file {} already exists".format(filename)
            )
    else:
        save(filename=filename, obj=obj)


# NOTE:
def load_obj(directory, filename):

    if filename.endswith(".pickle"):
        filename_sanitized = string_utils.remove_substring_from_end_of_string(
            string=filename, substring=".pickle"
        )
    else:
        filename_sanitized = filename

    filename = "{}/{}.pickle".format(
        directory, filename_sanitized
    )

    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
    else:
        raise IOError(
            "The file {} does not exist".format(filename)
        )

    return obj


# NOTE:
class PickleFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        #print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            #print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            #print("done.", flush=True)
            idx += batch_size


# NOTE:
def save_obj_in_bytes(directory, filename, obj, overwrite=False):

    def save_in_bytes(filename, obj):
        with open(filename, "wb+") as f:
            pickle.dump(obj, PickleFile(f), protocol=pickle.HIGHEST_PROTOCOL)

    if filename.endswith(".pickle"):
        filename_sanitized = string_utils.remove_substring_from_end_of_string(
            string=filename, substring=".pickle"
        )
    else:
        filename_sanitized = filename
    filename = "{}/{}.pickle".format(
        directory, filename_sanitized
    )

    if os.path.isfile(filename):
        if overwrite:
            save_in_bytes(filename=filename, obj=obj)
        else:
            print(
                "The file {} already exists".format(filename)
            )
    else:
        save_in_bytes(filename=filename, obj=obj)


# NOTE:
def load_obj_in_bytes(directory, filename):

    def load_in_bytes(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(PickleFile(f))

    if filename.endswith(".pickle"):
        filename_sanitized = string_utils.remove_substring_from_end_of_string(
            string=filename, substring=".pickle"
        )
    else:
        filename_sanitized = filename
    filename = "{}/{}.pickle".format(
        directory, filename_sanitized
    )

    if os.path.isfile(filename):
        load_in_bytes(filename=filename)
    else:
        raise IOError(
            "The file {} does not exist".format(filename)
        )

    return obj

if __name__ == "__main__":

    pass
