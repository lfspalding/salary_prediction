import csv
import gender_guesser.detector as gd
import pandas as pd

def get_gender(row, d):
    gender = 'u'  # 0 is male, 1 is female, 2 is undetermined
    name = row.split()
    gtemp = d.get_gender(name[0])
    if gtemp == 'male' or gtemp == 'mostly_male':
        gender = 'm'
        print(row)
    elif gtemp == 'female' or gtemp == 'mostly_female':
        gender = 'f'
    else:
        gender = 'u'
    return gender

def parse_gdata(input):
    d = gd.Detector(case_sensitive=False)
    with open(input) as csv_file:
        file_data = csv.reader(csv_file, delimiter=',')
        next(file_data)
        data = []
        for row in file_data:
            if row[2] != 'JobTitle' and row[2] != 'Not provided' and row[7] != '0':
                data.append([row[2], get_gender(row[1], d), row[3], row[7], row[9]])
    return data

gdata = parse_gdata('Salaries.csv')
#
# with open('cleaned_data.csv', 'w', newline='') as new_csv:
#     writer = csv.writer(new_csv)
#     writer.writerow(['Job','Gender','Base Salary','Total Pay','Year'])
#     writer.writerows(gdata)
