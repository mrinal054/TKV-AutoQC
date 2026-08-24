import pandas as pd
import os

def prep_df_multiCls(directories: list, labels: list, str_labels: list = None, 
                     ends_with: str = None, save_fullfile: str = None):

    """
    Prepares a pandas DataFrame containing file names, full paths, and class labels
    for a multi-class image classification task.

    :param directories: (list) List of directory paths, each corresponding to one class.
    :param labels: (list) List of integer class labels corresponding to each directory.
                   Must be the same length as `directories`.
    :param str_labels: (list) Optional list of string labels corresponding to each directory.
                       Must be the same length as `directories` if provided.
    :param ends_with: (str) Optional file extension or suffix to filter files (e.g., '.nii.gz').
                      If None, includes all files in each directory.
    :param save_fullfile: (str) Optional full path to save the resulting DataFrame as an Excel file.
                          If None, the file is not saved.

    :return df: (pandas.DataFrame) A DataFrame.
    """

    all_rows = []

    for i, dir in enumerate(directories):
        names = os.listdir(dir)

        if ends_with is not None:
            names = [name for name in names if name.endswith(ends_with)]

        fullfiles = [os.path.join(dir, name) for name in names]

        label_repeated = [labels[i]] * len(fullfiles)

        if str_labels is not None:
            str_label_repeated = [str_labels[i]] * len(fullfiles)
            for n, f, l, sl in zip(names, fullfiles, label_repeated, str_label_repeated):
                all_rows.append({"Names": n, "Directories": f, "Labels": l, "Str_labels": sl})
        else:
            for n, f, l in zip(names, fullfiles, label_repeated):
                all_rows.append({"Names": n, "Directories": f, "Labels": l})

    df = pd.DataFrame(all_rows)

    if save_fullfile is not None:
        df.to_excel(save_fullfile, index=False)

    return df


if __name__ == "__main__":
    raise SystemExit(
        "Import prep_df_multiCls() and provide your own directory and output paths."
    )
