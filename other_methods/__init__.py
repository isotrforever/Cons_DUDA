# money = []
# with open("DP_ann_money.csv", "r") as rf:
#     csv_reader = csv.reader(rf)
#     raws = [[raw[1], raw[2]] for raw in csv_reader if raw[2] in ["1", "0"]]
# money = raws
#
# quality = []
# with open("DP_ann_quality.csv", "r") as rf:
#     csv_reader = csv.reader(rf)
#     raws = [[raw[0], raw[1]] for raw in csv_reader if raw[1] in ["pos", "neg"]]
# quality = raws
#
# with open("money.csv", "w") as wf:
#     csv_writer = csv.writer(wf)
#     for raw in money:
#         csv_writer.writerow(raw)
#
# with open("quality.csv", "w") as wf:
#     csv_writer = csv.writer(wf)
#     for raw in quality:
#         if raw[1] == "pos":
#             csv_writer.writerow([raw[0], 1])
#         else:
#             csv_writer.writerow([raw[0], 0])
