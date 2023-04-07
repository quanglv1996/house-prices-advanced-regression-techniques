import os
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from util.reader import Reader
from util.util import create_folder
path_preprocess_data = './preprocessing'
create_folder(path_preprocess_data)

# Folder chứa thông tin về bộ dữ liệu MIMIC-III
path_dataset = '/home/quanglv/Downloads/MIMIC-III/'

# Khởi tạo đối tượng Reader cho việc đọc các file csv trong bộ dữ liệu MIMIC-III
reader = Reader(path_dataset=path_dataset)

'''
Định nghĩa công thức xác định bệnh nhân suy thận mãn (CKD) dựa trên eGFR (mức lọc cầu thận).
Công thức tính eGFR dựa trên tuổi, giới tính, chủng tộc và SCr của bệnh nhân tại thời điểm xét nghiệm.
'''


def caculate_eGFR(cr, gender, eth, age):
    temp = 186 * (cr ** (-1.154)) * (age ** (-0.203))
    if (gender == 'F'):
        temp = temp * 0.742
    if eth == 'BLACK/AFRICAN AMERICAN':
        temp = temp * 1.21
    return temp


'''
Do hầu hết bệnh nhân không có đủ thông tin về SCr 7 ngày trước khi nhập ICU.
Vì thế không thể xác định baseline SCr cụ thể cho từng bệnh nhân.
Sử dụng Baseline được định nghĩa theo KDIGO dựa trên tuổi, giới tính và chủng tộc người.
'''


def condition_kdigo(gender, eth, age):
    base = 0
    if (age >= 18) and (age <= 24):
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.5
            else:
                base = 1.3
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.2
            else:
                base = 1.0
    elif (age >= 25) and (age <= 29):
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.5
            else:
                base = 1.2
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.1
            else:
                base = 1.0
    elif (age >= 30) and (age <= 39):
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.4
            else:
                base = 1.2
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.1
            else:
                base = 0.9
    elif (age >= 40) and (age <= 54):
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.3
            else:
                base = 1.1
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.0
            else:
                base = 0.9
    elif (age >= 55) and (age <= 65):
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.3
            else:
                base = 1.1
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.0
            else:
                base = 0.8
    else:
        if (gender == 'M'):
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 1.2
            else:
                base = 1.0
        else:
            if (eth == 'BLACK/AFRICAN AMERICAN'):
                base = 0.9
            else:
                base = 0.8
    return base


'''
-----------------------------------------------------------------------------------------------------
|                               SÀNG LỌC BỆNH NHÂN AKI THỎA MÃN ĐIỀU KIỆN                           |
-----------------------------------------------------------------------------------------------------
'''

'''
Lấy thông tin về các lần nhập viện (ICUSTAY_ID, HADM_ID)thỏa mãn điều kiện
    + lưu vào file ADMISSIONS.csv
'''


def get_info_admissions():
    # Load các bảng PATIENTS và ICUSTAYS
    df = reader.read_admissions_table()

    # Lấy những bệnh nhân có thời gian nằm viện lớn hơn 72 giờ
    df['LOS_A'] = df['DISCHTIME'] - df['ADMITTIME']
    df['LOS_A'] = df['LOS_A'] / np.timedelta64(1, 'h')
    df = df[~(df['LOS_A'] < 72.0)]

    # Tính tuổi của bệnh nhân
    patients = reader.read_patients_table()
    df = pd.merge(df, patients, how='left', on='SUBJECT_ID')
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['AGE'] = (df['ADMITTIME'].dt.year - df['DOB'].dt.year)

    # Thay thế những bệnh nhân có số tuổi nhỏ hơn 0 bằng 90
    df.loc[((df.AGE > 89) | (df.AGE < 0)), 'AGE'] = 90
    df = df[(df['AGE'] >= 18)].sort_values(by=['SUBJECT_ID', 'ADMITTIME'], ascending=True)

    # Tính toán thời gian các bệnh nhân nhập ICU
    icustays = reader.read_icustay_table()
    df = pd.merge(df, icustays, how='right', on='HADM_ID')

    # Thời điểm vào ICU
    df['Time go ICU'] = (df['INTIME'] - df['ADMITTIME']) / np.timedelta64(1, 'h')

    # Thời gian sau khi nhập ICU
    df['Time after go ICU'] = (df['DISCHTIME'] - df['INTIME']) / np.timedelta64(1, 'h')

    # Số lần nhập ICU
    df['Count times go ICU'] = df.groupby('HADM_ID')['ICUSTAY_ID'].transform('count')

    # Lấy những bệnh nhân có thời gian vào ICU tính từ thời điểm vào lớn hơn 72 giờ
    # Vì dữ liệu cần đánh giá sử dụng 24 giờ và dựa vào 48 giờ tiếp theo để làm nhãn (xem bệnh nhân có bị AKI hay không)
    df = df[(df['Time after go ICU'] > 72.0)].drop_duplicates(subset=['ICUSTAY_ID']).sort_values(by=['SUBJECT_ID_x'])

    # Save thông tin của các lần nhập ICU thỏa mãn.
    with open(os.path.join(path_preprocess_data,'ADMISSIONS.csv'), 'a') as f:
        df.to_csv(f, encoding='utf-8', header=True)


'''
Lấy toàn bộ thông tin về xét nghiệm SCr của các bệnh nhân lớn hơn 18 tuổi và có thời gian nằm viện lớn hơn 72 giờ,
và thời gian nằm tại ICU lớn hơn 72 giờ.
Thông tin này sử dụng để sàng lọc các bệnh nhân bị CKD, AKI và thỏa mãn các điều kiện.
    + Lưu vào file C_LABEVENTS.csv
'''


def get_SCr_test():
    src_file = os.path.join(path_dataset, 'LABEVENTS.csv')
    hadm_id = pd.read_csv(os.path.join(path_preprocess_data,'ADMISSIONS.csv'))
    hadm_id = hadm_id.drop_duplicates(subset=['HADM_ID'])['HADM_ID']
    chunksize = 1 * (10 ** 6)
    print('Save information for each time in icu of patients...')
    list = pd.DataFrame()
    for chunk in pd.read_csv((src_file), header=0, index_col=0, chunksize=chunksize):
        chunk = chunk[chunk['ITEMID'] == 50912]
        list = list.append(chunk)
    list = list[list['HADM_ID'].isin(hadm_id)]
    print(list.shape)

    with open(os.path.join(path_preprocess_data,'C_LABEVENTS.csv'), 'a') as f:
        list.to_csv(f, encoding='utf-8', header=True)


'''
Sàng lọc bệnh nhân AKI, CDK, AKI trước khi nhập ICU và bình thường dựa trên thông tin về chỉ số SCr 
TYPE {1:AKI, 2:NORMAL, 3:CKD, 4:AKI_before }
    + Lưu vào file INFO_DATASET.csv
'''


def filter_aki_patients():
    c_lab = pd.read_csv(os.path.join(path_preprocess_data,'C_LABEVENTS.csv'))
    admissions = pd.read_csv(os.path.join(path_preprocess_data,'ADMISSIONS.csv'))

    df = pd.merge(admissions, c_lab, how='right', on='HADM_ID')
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['INTIME'] = pd.to_datetime(df['INTIME'])

    # Lấy thời điểm các xét nghiệm được thực hiện và ghi lại tính bắt đầu từ thời điểm nhập viện (HADM_ID)
    df['Time'] = (df['CHARTTIME'] - df['ADMITTIME']) / np.timedelta64(1, 'h')
    df = df[~df['Time'].isnull()]

    # Chuẩn hóa lại các giá trị SCr lỗi.
    df.loc[df['VALUENUM'] > 40, 'VALUENUM'] = (df[df['VALUENUM'] > 40]['VALUENUM'] / 88.4)
    df = df.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])

    # Lưu trữ thông tin cho từng lần nhập ICU.
    info_save = df.drop_duplicates(subset=['ICUSTAY_ID'])

    # TYPE {1:AKI, 2:NORMAL, 3:CKD, 4:AKI_before }
    info_save['TYPE'] = -1
    icustays_data = [frame for season, frame in df.groupby(['ICUSTAY_ID'])]

    # Biến đếm số lượng cho mỗi TYPE
    count_normal = 0  # Bệnh nhân bình thường
    count_aki = 0  # Bệnh nhân mắc AKI
    count_aki_before = 0  # Bệnh nhân mắc AKI trước khi nhập viện
    count_ckd = 0  # Bệnh nhân mắc CKD
    for temp in icustays_data:
        # Xử lý thông tin cho 1 lần nhập ICU
        temp = temp.sort_values(by=['Time'])

        # Lấy thời gian nhập ICU làm mốc để lọc các khoảng thời gian tiếp theo.
        # Các khoảng thời gian cần sử dụng bao gồm 72 giờ sau khi nhập ICU
        # (24 giờ cho dự đoán và 48 giờ cho việc xác định khi nhập ICU có bị AKI hay không)
        # Với những bệnh nhân có khoảng thời gian trước khi nhập ICU, ta cũng cần xác định bệnh nhân có bị AKI trước khi nhập ICU hay không.
        time_go_icu = temp['Time go ICU'].values[0]

        # Lấy khoảng thời gian tối đa 48 giờ (nếu có) của những bệnh nhân trước khi nhập ICU.
        data_time_before_go_icu = temp[(temp['Time'] < time_go_icu) & (temp['Time'] >= time_go_icu - 48)]

        # Lấy khoảng thời gian nằm ICU (24 giờ đầu), sử dụng cho dự đoán.
        data_24h_icu = temp[(temp['Time'] >= time_go_icu) & (temp['Time'] <= time_go_icu + 24)]

        # Lấy khoảng thời gian sau khi nhập ICU từ 24 đến 72 giờ, khảng thời gian sử dụng để đánh giá xem bệnh nhân có bị AKI khi nhập ICU hay không.
        data_48_to_72 = temp[(temp['Time'] > time_go_icu + 24) & (temp['Time'] <= time_go_icu + 72)]

        subject_id = temp['SUBJECT_ID_x'].values[0]

        # Loại bỏ những bệnh nhân không có xét nghiệm SCr trong 24 giờ đầu nhập ICU và 48 giờ tiếp theo.
        if data_24h_icu.empty or data_48_to_72.empty:
            continue

        # Tính toán SCr-min trong 24 giờ đầu
        cr = data_24h_icu['VALUENUM'].values.min()

        # Tính toán SCr max trong 48 giờ tiếp theo
        max72 = data_48_to_72['VALUENUM'].values.max()

        # Lấy thông tin về tuổi, giới tính và chủng tộc người.
        gender = data_24h_icu['GENDER'].values[0]
        age = data_24h_icu['AGE'].values[0]
        eth = data_24h_icu['ETHNICITY'].values[0]

        # Tính toán baseline và eGRF để xác định bệnh nhân AKI và CKD khi nhập ICU
        baseline = condition_kdigo(gender=gender, age=age, eth=eth)
        eGRF = caculate_eGFR(cr=cr, gender=gender, age=age, eth=eth)

        # Với những lần nhập ICU bệnh nhân có chỉ số eGRF dưới 60 thì những bệnh nhân này bị suy thận mãn (CKD) - Loại
        if eGRF < 60.0:
            print('SUBJECT_ID: {} | MIN_24: {} | MAX_72: {} | eGRF: {} | Standard: {} | SUY THAN MAN'.format(
                subject_id, cr, max72, eGRF, baseline))
            # Lưu lần nhập viện với TYPE = 3
            info_save.loc[df['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'TYPE'] = 3
            count_ckd = count_ckd + 1
        else:
            # Vì chúng ta muốn dự đoán AKI trước khi nó xảy ra vậy nên những bệnh nhân bị AKI trong 24 giờ đầu sẽ bị loại bỏ.
            # Nếu giá trị SCr max trong 48 giờ (từ 24 đến 71) lớn hơn giá trị SCr-min trong 24 giờ đầu + 0.3 - AKI
            # Nếu giá trị của SCr-max lớn hơn 1.5 lần baseline - AKI
            # (Dựa trên định nghĩa của KDIGO)
            if max72 > (cr + 0.3) or (max72 * 1.0) > (1.5 * baseline):
                # Kiểm tra xem những bệnh nhân mắc AKI trước khi nhập ICU hay không nếu có thông tin trước khi vào ICU.
                if not data_time_before_go_icu.empty:
                    min_before_go_icu = data_time_before_go_icu['VALUENUM'].values.min()
                    max24 = data_24h_icu['VALUENUM'].values.max()
                    if max24 > (min_before_go_icu + 0.3) or (max24 * 1.0) > (1.5 * baseline):
                        print(
                            'SUBJECT_ID: {} | MIN_24: {} | MAX_72: {} | eGRF: {} | Standard: {} | SUY THAN CAP TRUOC KHI VAO ICU'
                                .format(subject_id, cr, max72, eGRF, baseline))
                        info_save.loc[df['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'TYPE'] = 4
                        count_aki_before = count_aki_before + 1
                        continue
                else:
                    # Nếu không có thông tin trước khi nhập ICU thì sử dụng baseline.
                    max24 = data_24h_icu['VALUENUM'].values.max()
                    if (max24 * 1.0) > 1.5 * baseline:
                        print(
                            'SUBJECT_ID: {} | MIN_24: {} | MAX_72: {} | eGRF: {} | Standard: {} | SUY THAN CAP TRUOC KHI VAO ICU'
                                .format(subject_id, cr, max72, eGRF, baseline))
                        info_save.loc[df['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'TYPE'] = 4
                        count_aki_before = count_aki_before + 1
                        continue
                print('SUBJECT_ID: {} | MIN_24: {} | MAX_72: {} | eGRF: {} | Standard: {} | SUY THAN CAP'
                      .format(subject_id, cr, max72, eGRF, baseline))
                info_save.loc[df['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'TYPE'] = 1
                count_aki = count_aki + 1
            else:
                print('SUBJECT_ID: {} | MIN_24: {} | MAX_72: {} | eGRF: {} | Standard: {} | BINH THUONG'
                      .format(subject_id, cr, max72, eGRF, baseline))
                info_save.loc[df['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'TYPE'] = 2
                count_normal = count_normal + 1
    print('NORMAL: {}'.format(count_normal))
    print('AKI: {}'.format(count_aki))
    print('CKD: {}'.format(count_ckd))
    print('AKI before: {}'.format(count_aki_before))

    with open(os.path.join(path_preprocess_data,'INFO_DATASET.csv'), 'a') as f:
        info_save.to_csv(f, encoding='utf-8', header=True)


# NORMAL: 25393
# AKI: 2651
# CKD: 13826
# AKI before: 412

'''
Hiển thị thông tin về Dataset đang sử dụng (Option)
'''


def get_info_dataset():
    info = pd.read_csv('INFO_DATASET.csv')
    info = info[info['TYPE'].isin((1, 2))]
    info['ADMITTIME'] = pd.to_datetime(info['ADMITTIME'])
    info['CHARTTIME'] = pd.to_datetime(info['CHARTTIME'])
    info['INTIME'] = pd.to_datetime(info['INTIME'])

    info['24h in ICU'] = info['INTIME'] + pd.Timedelta(hours=24)
    info['24h in ICU'] = pd.to_datetime(info['24h in ICU'])
    info = info.astype({'AGE': 'int32'})
    list_dict_age = []
    for age in range(18, 91):
        num = info[info['AGE'] == age].shape[0]
        dict_age = {'AGE': age, 'NUMBER PATIENTS': num}
        list_dict_age.append(dict_age)
    age_statistical = pd.DataFrame(list_dict_age)
    print('AGE STATISTICAL:')
    print(age_statistical)
    print()
    print('SUBJECT_ID: {}'.format(info.drop_duplicates(subset=['SUBJECT_ID']).shape))
    print()
    print('SUBJECT_ID MALE: {}'.format(info[info['GENDER'] == 'M'].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print('SUBJECT_ID FEMALE: {}'.format(info[info['GENDER'] == 'F'].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print()
    print('SUBJECT_ID BLACK/AFRICAN AMERICAN: {}'.format(
        info[info['ETHNICITY'] == 'BLACK/AFRICAN AMERICAN'].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print('SUBJECT_ID OTHER: {}'.format(
        info[~(info['ETHNICITY'] == 'BLACK/AFRICAN AMERICAN')].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print()
    print('SUBJECT_ID AKI: {}'.format(info[info['TYPE'] == 1].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print('SUBJECT_ID NORMAL: {}'.format(info[info['TYPE'] == 2].drop_duplicates(subset=['SUBJECT_ID']).shape))
    print()
    print('HADM_ID: {}'.format(info.drop_duplicates(subset=['HADM_ID']).shape))
    print('ICUSTAY_ID: {}'.format(info.shape))


'''
-----------------------------------------------------------------------------------------------------
|                               Xử lý thông tin LABEVENTS                                           |
-----------------------------------------------------------------------------------------------------
'''

'''
Lấy các thông tin LABEVENTS cần sử dụng và tách biệt thành từng lần nhập ICU
    + Lưu vào file LABEVENTS.csv
    + Lưu thông tin từng lần nhập ICU trong folder raw_data_labevents
'''


def get_labevents_data():
    # Lấy thông tin những bệnh nhân được gán nhãn AKI và bình thường theo định nghĩa.
    info = pd.read_csv(os.path.join(path_preprocess_data,'INFO_DATASET.csv'))
    info = info[info['TYPE'].isin((1, 2))]
    info['ADMITTIME'] = pd.to_datetime(info['ADMITTIME'])
    info['CHARTTIME'] = pd.to_datetime(info['CHARTTIME'])
    info['INTIME'] = pd.to_datetime(info['INTIME'])
    info['24h in ICU'] = info['INTIME'] + pd.Timedelta(hours=24)
    info['24h in ICU'] = pd.to_datetime(info['24h in ICU'])

    # Lưu tất cả thông tin sử dụng trong LABEVENTS table
    labevents = pd.DataFrame()

    # Lấy thông tin từ bảng LABEVENTS in MIMIC-III dataset
    src_file = os.path.join(path_dataset, 'LABEVENTS.csv')
    path = os.path.join(path_preprocess_data,'raw_data_labevents')
    create_folder(path)
    chunksize = 1 * (10 ** 6)
    i = 1
    for chunk in pd.read_csv((src_file), header=0, index_col=0, chunksize=chunksize):
        print(i)
        chunk = chunk[~chunk['VALUENUM'].isnull()]  # Loại bỏ các hàng có giá trị NaN
        chunk = chunk[chunk['HADM_ID'].isin(info['HADM_ID'])]  # Lấy các thông tin theo HADM_ID đã lọc trước đó

        # Nhóm các thông tin lab theo từng HADM_ID (mỗi HADM_ID chứa thông tin của nhiều ICUSTAY_ID)
        hadms_info = [frame for season, frame in chunk.groupby(['HADM_ID'])]
        for hadm_info in hadms_info:
            hadm_info['CHARTTIME'] = pd.to_datetime(hadm_info['CHARTTIME'])
            info['24h in ICU'] = pd.to_datetime(info['24h in ICU'])

            # Lấy mã HADM_ID
            hadm_id = info[info['HADM_ID'] == int(hadm_info['HADM_ID'].values[0])]

            # Tách các lần nhập ICU (ICUSTAY_ID) của mỗi lần nhập viện (HADM_ID)
            for icustay_id in hadm_id['ICUSTAY_ID'].values:
                icustayid_ = hadm_id[hadm_id['ICUSTAY_ID'] == int(icustay_id)]
                intime = icustayid_['INTIME'].iloc[0]
                icutime = icustayid_['24h in ICU'].iloc[0]

                # Lấy các giá trị trong khoảng thời gian 24 giờ khi nhập ICU
                icustay_info = hadm_info[(hadm_info['CHARTTIME'] >= intime) & (hadm_info['CHARTTIME'] <= icutime)]
                icustay_info['ICUSTAY_ID'] = int(icustay_id)
                labevents = labevents.append(icustay_info)

                # Lưu các thông tin cho từng lần nhập ICU (ICUSTAY_ID)
                path_save = os.path.join(path, str(int(icustay_id)) + '.csv')
                if (os.path.exists(path_save)):
                    with open(path_save, 'a') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=False)
                else:
                    with open(path_save, 'w') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=True)
        i = i + 1
    with open(os.path.join(path_preprocess_data,'LABEVENTS.csv'), 'a') as f:
        labevents.to_csv(f, encoding='utf-8', header=True)


'''
Thông kê missing của các ITEMID,để chọn top các ITEMID missing ít nhất để sử dụng.
    + Lưu vào file STATISTICAL_MISSING_ITEMS.csv
'''


def statistical_itemid_missing():
    itemids = pd.read_csv(os.path.join(path_dataset, 'D_LABITEMS.csv'))
    chunksize = 1 * (10 ** 6)
    list_chunk = pd.DataFrame()
    for chunk in pd.read_csv((os.path.join(path_preprocess_data,'LABEVENTS.csv')), header=0, index_col=0, chunksize=chunksize):
        # Xóa bỏ những phép đo (ITEMID) trùng lặp cho mỗi lần nhập ICU (ICUSTAY_ID) của bệnh nhân
        chunk = chunk.drop_duplicates(subset=['ICUSTAY_ID', 'ITEMID'])
        list_chunk = list_chunk.append(chunk)
    list_chunk = list_chunk.drop_duplicates(subset=['ICUSTAY_ID', 'ITEMID'])

    # Số lượng ICUSTAY_ID được sử dụng.
    num_icustay_id = list_chunk.drop_duplicates(subset=['ICUSTAY_ID']).shape[0]

    # Số lượng các ITEMID không được thực hiện trong suốt quá trinh nằm viện sẽ bị missing
    temps = [frame for season, frame in list_chunk.groupby(['ITEMID'])]
    list_dict = []
    for temp in temps:
        num = temp.shape[0]  # Số lượng ICUSTAY_ID thực hiện với mỗi xét nghiệm (ITEMID)
        itemid = temp['ITEMID'].values[0]  # Mã của ITEMID
        ratio_missing = (1 - num / num_icustay_id) * 100  # Tính tỉ lệ missing của các ITEMID
        dict = {'ITEMID': itemid, 'NUMBER ICUSTAY_ID EXIT': num, 'RATIO MISSING': ratio_missing}
        list_dict.append(dict)
    df = pd.DataFrame(list_dict)
    df = df.sort_values(by=['NUMBER ICUSTAY_ID EXIT'], ascending=False)
    df = pd.merge(df, itemids, on='ITEMID', how='left')
    with open(os.path.join(path_preprocess_data,'STATISTICAL_MISSING_ITEMS_LABEVENTS.csv'), 'a') as f:
        df.to_csv(f, encoding='utf-8', header=True)


'''
Chuyển các thông tin dạng raw sang dạng có nhãn thời gian, lưu thông tin về giá trị mean của các ICUSTAY_ID
    + Lưu thông tin vào folder ./timeseries_lab
'''


def convert_raw2timeseries_labevents():
    path =os.path.join(path_preprocess_data, 'raw_data_labevents')
    path_series = os.path.join(path_preprocess_data,'data_timelabel_labevents')
    create_folder(path_series)

    # Lấy tên các file csv
    filenames = os.listdir(path)

    # Sử dụng các xét nghiệm (ITEMID) có tỉ lệ missing dưới 50%
    items = pd.read_csv(os.path.join(path_preprocess_data,'STATISTICAL_MISSING_ITEMS_LABEVENTS.csv'))
    items = items.sort_values(by=['ITEMID'])
    label_items = list((items['ITEMID'].values))
    label_items.append('Time')

    for filename in filenames:
        print(filename)
        timeseries_raw = []
        df = pd.read_csv(os.path.join(path, filename))
        df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
        df = df[df['ITEMID'].isin(items['ITEMID'])]  # Lấy các ITEMID được sử dụng
        # Loại bỏ những xét nghiệm trong cùng 1 mốc thời gian, tránh xung đột.
        df = df.sort_values(by=['CHARTTIME'], ascending=True).drop_duplicates(subset=['ITEMID', 'CHARTTIME'])

        # Lấy các mốc thời gian thực hiện các phép đo
        list_time = df.drop_duplicates(subset=['CHARTTIME'], keep='first').sort_values(by=['CHARTTIME'],
                                                                                       ascending=True).CHARTTIME
        # Lấy thời gian bắt đầu của lần đo đầu tiên trong ICU
        start_time = list_time.iloc[0]
        timeseries_raw.append(label_items)

        # Xử lý các xét nghiệm trên từng mốc thời gian
        for i in range(list_time.size):
            # Lấy thông tin các xét nghiệm theo từng mốc thời gian
            df_sub = df[df.CHARTTIME == list_time.iloc[i]]

            time_ = pd.Series((list_time.iloc[i] - start_time) / pd.Timedelta('1 hour')).values[0]
            df_sub = pd.merge(items, df_sub, on='ITEMID', how='left')
            df_sub = df_sub.sort_values(by=['ITEMID'])
            values_raw = list(df_sub.VALUENUM.values)

            # Thêm thông tin tại mỗi bước thời gian cho tất cả các items. Những Items không tồn tại được gán NaN
            values_raw.append(time_)
            timeseries_raw.append(values_raw)
        timeseries_raw = pd.DataFrame(timeseries_raw, columns=timeseries_raw.pop(0))
        with open(os.path.join(path_series, filename), 'a') as f:
            timeseries_raw.to_csv(f, encoding='utf-8', header=True)


'''
-----------------------------------------------------------------------------------------------------
|                               Xử lý thông tin CHARTEVENTS                                         |
-----------------------------------------------------------------------------------------------------
'''

'''
Xử lý thông tin của bảng CHARTEVENTS sử dụng 
    + Lưu vào file CHARTEVENTS.csv
    + Thông tin về mỗi lần nhập ICU được lưu trữ trong folder ./raw_data_chartevents
'''


def get_chartevents_data():
    info = pd.read_csv(os.path.join(path_preprocess_data,'INFO_DATASET.csv'))
    info = info[info['TYPE'].isin((1, 2))]
    info['ADMITTIME'] = pd.to_datetime(info['ADMITTIME'])
    info['CHARTTIME'] = pd.to_datetime(info['CHARTTIME'])
    info['INTIME'] = pd.to_datetime(info['INTIME'])
    info['24h in ICU'] = info['INTIME'] + pd.Timedelta(hours=24)
    info['24h in ICU'] = pd.to_datetime(info['24h in ICU'])
    src_file = os.path.join(path_dataset, 'CHARTEVENTS.csv')

    path = os.path.join(path_preprocess_data,'raw_data_chartevents')
    create_folder(path)
    chunksize = 5 * (10 ** 6)
    i = 1
    for chunk in pd.read_csv((src_file), header=0, index_col=0, chunksize=chunksize):
        print(i)
        chunk = chunk[~(chunk['ERROR'] == 1)]
        chunk = chunk[~chunk['VALUENUM'].isnull()]
        chunk = chunk[chunk['HADM_ID'].isin(info['HADM_ID'])]
        hadms_info = [frame for season, frame in chunk.groupby(['HADM_ID'])]
        for hadm_info in hadms_info:
            hadm_info['CHARTTIME'] = pd.to_datetime(hadm_info['CHARTTIME'])
            info['24h in ICU'] = pd.to_datetime(info['24h in ICU'])
            hadm_id = info[info['HADM_ID'] == int(hadm_info['HADM_ID'].values[0])]
            for icustay_id in hadm_id['ICUSTAY_ID'].values:
                icustayid_ = hadm_id[hadm_id['ICUSTAY_ID'] == int(icustay_id)]
                intime = icustayid_['INTIME'].iloc[0]
                icutime = icustayid_['24h in ICU'].iloc[0]
                icustay_info = hadm_info[(hadm_info['CHARTTIME'] >= intime) & (hadm_info['CHARTTIME'] <= icutime)]
                icustay_info['ICUSTAY_ID'] = int(icustay_id)

                # Lưu các thông tin cho từng lần nhập ICU (ICUSTAY_ID)

                path_save = os.path.join(path, str(int(icustay_id)) + '.csv')
                if (os.path.exists(path_save)):
                    with open(path_save, 'a') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=False)
                else:
                    with open(path_save, 'w') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=True)

                # Save to chart
                if (os.path.exists(os.path.join(path_preprocess_data,'CHARTEVENTS.csv'))):
                    with open(os.path.join(path_preprocess_data,'CHARTEVENTS.csv'), 'a') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=False)
                else:
                    with open(os.path.join(path_preprocess_data,'CHARTEVENTS.csv'), 'w') as f:
                        icustay_info.to_csv(f, encoding='utf-8', header=True)
        i = i + 1


'''
Thông kê missing của các ITEMID, chọn top các ITEMID missing ít nhất để sử dụng
Sau khi thực hiện thống kê sẽ thực hiện lựa chọn bằng tay các ITEMID 
    + lưu ra file TOP_CHART_ITEMS.csv
'''


def statistical_chart_itemid_missing():
    mapping_itemids = pd.read_csv(os.path.join(path_preprocess_data,'itemid_to_variable_map.csv'))
    mapping_itemids = mapping_itemids[['LEVEL1', 'STATUS', 'ITEMID', 'MIMIC LABEL', 'CATEGORY', 'FLUID']]
    mapping_itemids.loc[mapping_itemids.STATUS == 'verify', 'MIMIC LABEL'] = mapping_itemids[
        mapping_itemids.STATUS == 'verify'].LEVEL1
    mapping_itemids.loc[mapping_itemids.STATUS == 'ready', 'MIMIC LABEL'] = mapping_itemids[
        mapping_itemids.STATUS == 'ready'].LEVEL1

    chunksize = 5 * (10 ** 6)
    list_chunk = pd.DataFrame()
    for chunk in pd.read_csv((os.path.join(path_preprocess_data,'CHARTEVENTS.csv')), header=0, index_col=0, chunksize=chunksize):
        chunk = chunk.drop_duplicates(subset=['ICUSTAY_ID', 'ITEMID'])
        list_chunk = list_chunk.append(chunk)
    list_chunk = pd.merge(list_chunk, mapping_itemids, on='ITEMID', how='left')
    list_chunk = list_chunk.drop_duplicates(subset=['ICUSTAY_ID', 'MIMIC LABEL'])

    num_icustay_id = list_chunk.drop_duplicates(subset=['ICUSTAY_ID']).shape[0]
    temps = [frame for season, frame in list_chunk.groupby(['MIMIC LABEL'])]
    list_dict = []
    for temp in temps:
        num = temp.shape[0]
        item_label = temp['MIMIC LABEL'].values[0]
        ratio_missing = (1 - num / num_icustay_id) * 100
        dict = {'MIMIC LABEL': item_label, 'NUMBER ICUSTAY_ID EXIT': num, 'RATIO MISSING': ratio_missing}
        list_dict.append(dict)
    df = pd.DataFrame(list_dict)
    df = df.sort_values(by=['NUMBER ICUSTAY_ID EXIT'], ascending=False)
    df = pd.merge(df, mapping_itemids, on='MIMIC LABEL', how='right')
    with open(os.path.join(path_preprocess_data,'STATISTICAL_MISSING_ITEMS_CHARTEVENTS.csv'), 'a') as f:
        df.to_csv(f, encoding='utf-8', header=True)


'''
Chuyển thông tin trong chartevents từ dạng raw sang dạng timeseries
    + Lưu ra folder ./timeseries_chart
'''


def convert_raw2timeseries_chartevents():
    path = os.path.join(path_preprocess_data,'./raw_data_chartevents')
    path_series = os.path.join(path_preprocess_data,'./data_timelabel_chartevents')
    create_folder(path_series)
    filenames = os.listdir(path)

    items = pd.read_csv(os.path.join(path_preprocess_data,'TOP_CHART_ITEMS.csv'))
    items = items.sort_values(by=['RATIO MISSING'])

    label_items = list((items.drop_duplicates(subset=['MIMIC LABEL'])['MIMIC LABEL']).values)
    print(label_items)
    label_items.append('Time')
    for filename in filenames:
        print(filename)
        timeseries_raw = []
        df = pd.read_csv(os.path.join(path, filename))
        df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
        df = df[df['ITEMID'].isin(items['ITEMID'])]
        df = df.sort_values(by=['CHARTTIME'], ascending=True).drop_duplicates(subset=['ITEMID', 'CHARTTIME'])
        if df.empty:
            print('Data empty!')
            continue
        # Get time
        list_time = df.drop_duplicates(subset=['CHARTTIME'], keep='first').sort_values(by=['CHARTTIME'],
                                                                                       ascending=True).CHARTTIME
        # Start time
        start_time = list_time.iloc[0]
        timeseries_raw.append(label_items)

        # Processing data to time-series
        for i in range(list_time.size):
            # Devide Data Frame follow time (CHARTTIME)
            df_sub = df[df.CHARTTIME == list_time.iloc[i]]

            # Convert yyyy:mm:hh to hh
            time_ = pd.Series((list_time.iloc[i] - start_time) / pd.Timedelta('1 hour')).values[0]
            df_sub = pd.merge(items, df_sub, on='ITEMID', how='left')
            df_sub = df_sub.sort_values(by=['RATIO MISSING', 'MIMIC LABEL', 'VALUENUM'])
            df_sub = df_sub.drop_duplicates(subset=['MIMIC LABEL'], keep='first')
            values_raw = list(df_sub.VALUENUM.values)

            # Thêm thông tin tại mỗi bước thời gian cho tất cả các items. Những Items không tồn tại được gán NaN
            values_raw.append(time_)
            timeseries_raw.append(values_raw)
        timeseries_raw = pd.DataFrame(timeseries_raw, columns=timeseries_raw.pop(0))

        with open(os.path.join(path_series, filename), 'a') as f:
            timeseries_raw.to_csv(f, encoding='utf-8', header=True)


'''
-----------------------------------------------------------------------------------------------------
|                               CHIA DATA TRAINING AND TEST                                          |
-----------------------------------------------------------------------------------------------------
'''

'''
Chia tập training và tập test
    + Lưu tập training ra file DATA_TRAIN.csv
    + Lưu tập test ra file DATA_TEST.csv
'''


def divide_dataset():
    path_lab = os.path.join(path_preprocess_data,'./data_timelabel_labevents')
    lab_filenames = os.listdir(path_lab)
    path_chart = os.path.join(path_preprocess_data,'./data_timelabel_chartevents')
    chart_filenames = os.listdir(path_chart)
    lab_list_icutsayid = []
    chart_list_icustayid = []

    info_icustayid = pd.read_csv(os.path.join(path_preprocess_data,'INFO_DATASET.csv'))
    info_icustayid['LAB'] = 0
    info_icustayid['CHART'] = 0
    info_icustayid['USE'] = 0

    #Kiểm tra xem mỗi lần nhập ICU có đủ thông tin về cả CHARTEVENTS và LABEVENTS không
    #Kiểm tra LAB
    for filename in lab_filenames:
        icustayid = int(filename.split('.')[0])
        lab_list_icutsayid.append(icustayid)
    info_icustayid.loc[info_icustayid['ICUSTAY_ID'].isin(lab_list_icutsayid), 'LAB'] = 1

    #Kiểm tra CHART
    for filename in chart_filenames:
        icustayid = int(filename.split('.')[0])
        chart_list_icustayid.append(icustayid)
    info_icustayid.loc[info_icustayid['ICUSTAY_ID'].isin(chart_list_icustayid), 'CHART'] = 1

    #Kiểm tra cả 2
    info_icustayid.loc[(info_icustayid['LAB'] == 1) & (info_icustayid['CHART'] == 1), 'USE'] = 1
    info_icustayid_ = info_icustayid[info_icustayid['USE'] == 1]

    # Lấy thông tin về lần nhập ICU bị AKI và bình thường
    aki = info_icustayid_[info_icustayid_['TYPE'] == 1]
    normal = info_icustayid_[info_icustayid_['TYPE'] == 2]

    #Tỉ lệ chia tập trianing và tập test
    ratio = 0.7

    aki_train = aki.head(int(ratio * aki.shape[0]))
    normal_train = normal.head(int(ratio * normal.shape[0]))

    train_set = pd.concat([aki_train, normal_train])
    with open('DATA_TRAIN.csv', 'a') as f:
        train_set.to_csv(f, encoding='utf-8', header=True)
    test_set = info_icustayid_[~(info_icustayid_['ICUSTAY_ID'].isin(train_set.ICUSTAY_ID))]
    with open('DATA_TEST.csv', 'a') as f:
        test_set.to_csv(f, encoding='utf-8', header=True)


'''
Chạy chương trình
'''
def run():
    get_info_admissions()
    get_SCr_test()
    filter_aki_patients()
    get_labevents_data()
    statistical_itemid_missing()
    convert_raw2timeseries_labevents()
    get_chartevents_data()
    statistical_chart_itemid_missing()
    convert_raw2timeseries_chartevents()
    divide_dataset()

run()

