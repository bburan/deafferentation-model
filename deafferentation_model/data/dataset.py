import pandas as pd

from .. import poles


class Dataset:

    def load_poles(self):
        raise NotImplementedError

    def load_nh_poles(self):
        return pd.DataFrame([poles.load_nh_poles()], index=['NH'])
