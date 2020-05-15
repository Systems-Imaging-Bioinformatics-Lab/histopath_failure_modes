"""Functions to parse outputs of DeepPath results
"""
import re
def parse_auc_file(path):
    """Parse my amalgimation of the AUC summaries (auc_summary.txt)
    """
    entries = []
    with open(path,"r") as f:
        for line in f.readlines():
            entry = {}
            fields = line.split()
            entry["type"] = "tile" if fields[0]=="out1" else "slide"
            if fields[4] == "c1auc":
                entry["class"] = "Normal"
            elif fields[4] == "c2auc":
                entry["class"] = "LUAD"
            else:
                entry["class"] = "LUSC"
            entry["auc"] = float(fields[5])
            entry["ci_lower"] = float(fields[7])
            entry["ci_upper"] = float(fields[8])
            entry["best_thresh"] = float(fields[9][1:])
            entries.append(entry)
    return entries
def parse_slide_probs(path):
    """Parse per slide probabilities (out2PerSlideStats.txt)
    """
    entries = []
    with open(path,"r") as f:
        for line in f.readlines():
            entry = {}
            fields = line.split("\t")
            entry["name"] = re.findall("test_(.*)\.",fields[0])[0]
            true_labels= re.findall(r"\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)]",fields[1])[0]
            true_labels = [float(x) for x in true_labels]
            pos_label = max(range(len(true_labels)),key=lambda x: true_labels[x])
            if pos_label == 0.0:
                entry["class"] = "Normal"
            elif pos_label == 1.0:
                entry["class"] = "LUAD"
            else:
                entry["class"] = "LUSC"
            percent_select = [
                float(re.findall("\d+.\d+",fields[2])[0]),
                float(fields[3]),
                float(fields[4])
            ]
            av_prob = [
                float(re.findall("\d+.\d+",fields[5])[0]),
                float(fields[6]),
                float(fields[7])
            ]
            entry["true_percent_select"] = percent_select[pos_label]
            entry["true_av_prob"] = av_prob[pos_label]
            entry["normal_prob"] = av_prob[0]
            entry["luad_prob"] = av_prob[1]
            entry["lusc_prob"] = av_prob[2]
            entry["normal_selected"] = percent_select[0]
            entry["luad_selected"] = percent_select[1]
            entry["lusc_selected"] = percent_select[2]
            entry["ntiles"] = float(re.findall("\d+.\d+",fields[8])[0])
            entries.append(entry)
    return entries
def parse_tile_probs(path):
    """Parse tile level probabilities (out_filename_Stats.txt)
    """
    entries = []
    with open(path,"r") as f:
        for line in f.readlines():
            entry = {}
            fields = line.split("\t")
            name_parsed = re.findall(r"test_(.*)_(\d+_\d+).dat",fields[0])[0]
            entry["name"] = name_parsed[0]
            entry["tile"] = name_parsed[1]
            entry["correct_class"] = bool(fields[1])
            probs = re.findall(r".*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+).*",fields[2])[0][1:]
            entry["normal_prob"] = float(probs[0])
            entry["luad_prob"] = float(probs[1])
            entry["lusc_prob"] = float(probs[2])
            entry["corrected_true_prob"] = float(fields[3])
            true_class = int(fields[5])
            if true_class==1:
                entry["class"] = "Normal"
            elif true_class==2:
                entry["class"] = "LUAD"
            else:
                entry["class"] = "LUSC"
            entries.append(entry)
    return entries

def parse_modified_tile_log(path):
    """Parse log of modified tiles (log_modified_tiles.tsv)
    """
    entries = []
    with open(path,"r") as f:
        for line in f.readlines():
            fields = line.strip().split("/")
            entry = (fields[4][:-6],fields[7][:-5])
            entries.append(entry)
    return entries
