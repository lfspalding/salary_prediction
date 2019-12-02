import pandas as pd
import numpy as np


def pay_average(pd_data):
    f_base = 0
    m_base = 0
    u_base = 0
    f_total = 0
    m_total = 0
    u_total = 0
    f_count = 0
    m_count = 0
    u_count = 0
    count = 0
    for index, row in pd_data.iterrows():
        if row['Job'] == 'SPECIAL NURSE':
            if row['Gender'] == "m":
                temp = row['Base Salary']
                if not pd.isna(temp):
                    m_base += temp
                else: print(temp)
                m_total += row['Total Pay']
                m_count += 1
            if row['Gender'] == "f":
                temp = row['Base Salary']
                if not pd.isna(temp):
                    f_base += temp
                f_total += row['Total Pay']
                f_count += 1
            else:
                temp = row['Base Salary']
                if not pd.isna(temp):
                    u_base += temp
                u_total += row['Total Pay']
                u_count += 1
    pay_avg = [['men', m_count, m_base/m_count, m_total/m_count],['women', f_count, f_base/f_count, f_total/f_count],
                ['other', u_count, u_base/u_count, u_total/u_count]]
    pd_pay_avg = pd.DataFrame(pay_avg, columns=['Gender', 'Number', 'Base Pay Avg', 'Total Pay Avg'])
    return pd_pay_avg


data = pd.read_csv('cleaned_data.csv', delimiter=',')
print(pay_average(data))


