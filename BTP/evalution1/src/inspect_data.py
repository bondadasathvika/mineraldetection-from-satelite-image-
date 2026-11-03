import os
import scipy.io as sio


def inspect_mat_file(filepath):

    print("=" * 70)
    print("FILE:", filepath)
    print("=" * 70)

    if not os.path.exists(filepath):
        print("ERROR: File does not exist")
        return

    try:
        data = sio.loadmat(filepath)

        for key, value in data.items():

            if key.startswith("__"):
                continue

            print("\nVariable:", key)
            print("Type:", type(value))

            try:
                print("Shape:", value.shape)
                print("Dtype:", value.dtype)
            except AttributeError:
                print("No shape information")

    except Exception as e:

        print("Could not read using scipy.io.loadmat")
        print("Error:", e)


if __name__ == "__main__":

    data_file = (
        "data/Cuprite/Cuprite_data_R188.mat"
    )

    gt_file = (
        "data/Cuprite/Cuprite_GT_nEnd12.mat"
    )

    inspect_mat_file(data_file)

    inspect_mat_file(gt_file)