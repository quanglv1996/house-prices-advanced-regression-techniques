import csv
import os


def write_log(file_name, list_of_dict):
    key = list_of_dict[0].keys()
    if os.path.isfile(file_name):
        with open(file_name, 'a') as output_file:
            dict_writer = csv.DictWriter(output_file, key)
            dict_writer.writerows(list_of_dict)
    else:
        with open(file_name, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, key)
            dict_writer.writeheader()
            dict_writer.writerows(list_of_dict)
