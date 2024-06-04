# Global paths
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


XVEC_SC = "../Results/Custom pipeline/xvec_sc"
XVEC_AHC = "../Results/Custom pipeline/xvec_ahc"

ECAPA_SC = "../Results/Custom pipeline/ecapa_sc"
ECAPA_AHC = "../Results/Custom pipeline/ecapa_ahc"


def main():
    # Go through the RTTM files and read one file at a time
    for path in [XVEC_SC, XVEC_AHC, ECAPA_SC, ECAPA_AHC]:
        for file in os.listdir(path):
            if file.endswith(".rttm"):
                # Read the file, specifying the <NA> values
                df = pd.read_csv(
                    os.path.join(XVEC_SC, file), sep=" ", header=None, na_values=["<NA>"]
                )

                # Modify column 2 and 7
                df[2] = 1  # Change column 2 to 1
                # Change column 7 to 'spk{number}' and add leading zeros if necessary
                df[7] = "spk" + df[7].astype(str)
                df[7] = df[7].apply(lambda x: x if len(x) >= 5 else x[:3] + "0" + x[3:])

                # Replace NaN with <NA>
                df = df.fillna("<NA>")

                # Write the modified dataframe back to the file
                df.to_csv(
                    os.path.join(XVEC_SC, file),
                    sep=" ",
                    header=None,
                    index=False,
                    na_rep="<NA>",
                    quoting=False,
                )

                print(f"Modified file: {file}")

                sys.exit()


if __name__ == "__main__":
    main()
