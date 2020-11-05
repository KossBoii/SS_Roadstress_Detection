import os
import csv

class APScore:
    def __init__(self, AP, AP50, AP75, APs, APm, APl):
        self.AP = AP
        self.AP50 = AP50
        self.AP75 = AP75
        self.APs = APs
        self.APm = APm
        self.APl = APl

#self.bbox_result = APScore(bbox_AP, bbox_AP50, bbox_AP75, bbox_APs, bbox_APm, bbox_APl)
#self.segm_result = APScore(segm_AP, segm_AP50, segm_AP75, segm_APs, segm_APm, segm_APl)
class Result:
    def __init__(self):
        self.threshold = threshold
        self.new_det_time = new_det_time
        self.old_det_time = old_det_time
        self.new_bbox_result = new_bbox_result
        self.new_segm_result = new_segm_result
        self.old_bbox_result = old_bbox_result
        self.old_segm_result = old_segm_result

class csvWriter:
    # field names
    fields = ["Model Folder", "Model ID", "Training Time", "MAX_ITER (k iterations)", "Backbone Model", "BATCH_SIZE_PER_IMAGE",
            "ANCHOR_GENERATOR.SIZES", "ANCHOR_GENERATOR.ANGLES", "BASE_LR", "WARMUP_ITERS", "IMS_PER_BATCH", "Total_loss",
            "Threshold", "old detection time (s)", "new detection time (s)", 
            "bbox_old_AP", "bbox_old_AP50", "bbox_old_AP75", "bbox_old_APs", "bbox_old_APm", "bbox_old_APl",
            "segm_old_AP", "segm_old_AP50", "segm_old_AP75", "segm_old_APs", "segm_old_APm", "segm_old_APl",
            "bbox_new_AP", "bbox_new_AP50", "bbox_new_AP75", "bbox_new_APs", "bbox_new_APm", "bbox_new_APl",
            "segm_new_AP", "segm_new_AP50", "segm_new_AP75", "segm_new_APs", "segm_new_APm", "segm_new_APl",]

    def __init__(self, list_of_results):
        # self.ID = ID                                        # model ID
        # self.mFolder = mFolder                              # model folder
        # self.tTime = tTime                                  # training time
        # self.iter = iter                                    # max training iterations
        # self.model = model                                  # backbone model
        # self.batch_size_per_img = batch_size_per_img        # batch size per image
        # self.anchor_sizes = anchor_sizes                    # anchor generator sizes
        # self.anchor_angles = anchor_angles                  # anchor generator angles
        # self.base_lr = base_lr                              # base learning rate
        # self.warmup_iter = warmup_iter                      # warmup iterations
        # self.img_per_batch = img_per_batch                  # image per batch
        # self.total_loss = total_loss                        # total loss
        self.list_of_results = list_of_results

    def getVal(self, input):
        return input[2][len(input[0]) + 1:]

    def write_csv(self, fileName):
        # Print stuffs to the file
        csvFile = open(fileName, 'w')
        w = csv.writer(csvFile)
        w.writerow(self.fields)
        
        f = open("output_532.txt")
        lines = f.readlines()

        # Start writing values
        num_models = int(len(self.list_of_results) / 55)
        count = 0

        for i in range(0, num_models):
            #for j in range(0, 55):
            row = []
            row.append(str(count))
            row.append(lines[list_of_results[i*55][1] - 4])
            for j in range(0, 10):
                row.append(self.getVal(list_of_results[i*55 + j]))
            
            rows = []
            for j in range(0, 9):
                temp = row.copy()
                temp.append(self.getVal(list_of_results[i*55 + 10 + 5*j]))
                temp.append(self.getVal(list_of_results[i*55 + 10 + 5*j + 1]))
                temp.append(self.getVal(list_of_results[i*55 + 10 + 5*j + 2]))

                rs_old_lNum = list_of_results[i*55 + 10 + 5*j + 4][1]
                rs_new_lNum = list_of_results[i*55 + 10 + 5*j + 3][1]
                
                # roadstress_old dataset:
                old_metrics_bbox = lines[rs_old_lNum + 3].rstrip("\n").split(":")[4][1:].split(",")
                old_metrics_segm = lines[rs_old_lNum + 6].rstrip("\n").split(":")[4][1:].split(",")
                temp += old_metrics_bbox
                temp += old_metrics_segm

                # roadstress_new dataset:
                new_metrics_bbox = lines[rs_new_lNum + 3].rstrip("\n").split(":")[4][1:].split(",")
                new_metrics_segm = lines[rs_new_lNum + 6].rstrip("\n").split(":")[4][1:].split(",")
                temp += new_metrics_bbox
                temp += new_metrics_segm

                rows.append(temp)

            w.writerows(rows)
            count += 1

        f.close()
        csvFile.close()
        
def search_multiple_strings_in_file(file_name, list_of_strings):
    """Get line from the file along with line numbers, which contains any string from the list"""
    line_number = -1
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            line_number += 1
            # For each line, check if line contains any string from the list of strings
            for string_to_search in list_of_strings:
                if string_to_search in line:
                    # If any string is found in line, then append that line along with line number in list
                    list_of_results.append((string_to_search, line_number, line.rstrip()))
    # Return list of tuples containing matched string, line numbers and lines where string is found
    return list_of_results


if __name__ == "__main__":
    list_of_strings = ["Training time:", "Max iterations:", "Model:", "Batch_size_per_img:",
                        "Anchor generator sizes:", "Anchor generator angles:", "Base LR:",
                        "Warmup Iter:", "IMS_PER_BATCH:", "Total Loss:", "Threshold:",
                        "Average Inferencing Time for roadstress_old dataset:",
                        "Average Inferencing Time for roadstress_new dataset:",
                        "Evaluation results for roadstress_new_val in csv format:",
                        "Evaluation results for roadstress_old_val in csv format:",
                        ]
    
    list_of_results = search_multiple_strings_in_file("output_532.txt", list_of_strings)
    f = csvWriter(list_of_results)
    f.write_csv("result.csv")

    







