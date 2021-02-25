"""
1 -
    Transform from xlsx to csv;
2 -
    Split para to stcs;
    Segmentation;
    Remove stop words;
3 -
    Build tokens;
    Rule based filter;
"""


import pandas as pd


if __name__ == '__main__':
    data_xls = pd.read_excel("source_data/0_xlsx/1.xlsx")
    data_xls.to_csv("1.csv", encoding="utf-8")
