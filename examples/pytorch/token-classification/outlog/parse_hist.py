import csv
content = open("hist_sta.txt").readlines()

i_layer = 0
w1_flag = 0
w1_dict = {}
w2_dict = {}
for line in content:
    for key in ["variance","skewness","kurtosis","tokenuni","clsUni"]:
        if key in line:
            if w1_flag >0:
                if key in w1_dict.keys():
                    w1_dict[key].append(line.strip().split(":")[-1])
                else:
                    w1_dict[key] = []
                    w1_dict[key].append(line.strip().split(":")[-1])
            else:
                if key in w2_dict.keys():
                    w2_dict[key].append(line.strip().split(":")[-1])
                else:
                    w2_dict[key] = []
                    w2_dict[key].append(line.strip().split(":")[-1])
    if "!!!" in line or "***" in line:
        w1_flag = (w1_flag+1)%2
print(w1_dict)
print()
print(w2_dict)

w1_file = open("w1_file.csv","w")
for key in w1_dict:
    w1_file.writelines(key+",")
    for item in w1_dict[key]:
        w1_file.writelines(item+",")
w1_file.close()

w2_file = open("w2_file.csv","w")
for key in w2_dict:
    w2_file.writelines(key+",")
    for item in w2_dict[key]:
        w2_file.writelines(item+",")
w2_file.close()