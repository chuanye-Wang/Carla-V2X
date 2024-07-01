'''
这部分代码还可以继续完善，可以将两个函数融合，完成多个x搜索，并将所有id的数据打包成新的csv生成出来

作者：王川页
日期：2024.6.28
邮件：553503540@qq.com
'''
import pandas as pd
import math
import random
import sys,os
import numpy

command = 4

if command == 1:
    '''
    @brief 此代码负责查找含有某一x值的object的ID的，从数据集csv文件当中
    '''
    try:
            path2dataset_file = "10039.csv"
            # print(f'dataset path: {path2dataset_file}')
            raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
            colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
            raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据
            # print(raw_data)

            ##########################################################################
            # 这一部分代码可以将某一个id的数据单独提取出来，这部分代码用于调试，最后注释掉即可
            #
            # some_x = 419518.201555984  # 长弯道的，也是我最倾向于选择的轨迹     2470283
            # some_x = 419529.565373655  # 长弯道的                             2468980
            # some_x = 419521.63902373   # 直线的                               2469077
            # some_x = 419543.576778995  # 短弯道的，第一个数据点就是             2445943
            some_x = 419546.032637883  # 短弯道的                             2466088

            found = raw_data[raw_data['x'] == some_x]
            print(f'找到符合的一行：\n{found}\n')

            found_id = int(found['id'].values)
            print(f'现在找到该轨迹点属于id为 {found_id} 的对象:\n')
            # raw_data = raw_data[raw_data['id'] == found_id]

            # print(f'现在将id为 {found_id} 的所有数据提取出来：')
            # print(raw_data,'\n')


    except Exception as e:
            print(f'选择函数出现错误：{e}')


if command == 2:
    '''
    @brief 用于将特定ID的所有数据全部抽取出来
    '''
    # 长弯道的，也是我最倾向于选择的轨迹     2470283
    # 长弯道的                             2468980
    # 直线的                               2469077
    # 短弯道的，第一个数据点就是             2445943
    # 短弯道的                             2466088
    aim_ID = [2470283, 2468980, 2469077]
    output_file = '10039_3_car.csv'


    path2dataset_file = "10039.csv"
    # print(f'dataset path: {path2dataset_file}')
    raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
    colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
    raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据

    
    new_data_frame = pd.DataFrame(columns=colums_we_want)

    for this_ID in aim_ID:
        all_data_of_this_ID = raw_data[raw_data['id'] == this_ID]
        new_data_frame = pd.concat([new_data_frame, all_data_of_this_ID], ignore_index=True)
    

    try:
        
        new_data_frame.to_csv(output_file, index=False)
        print(f'生成文件{output_file}成功')
    except Exception as e:
         print(f'生成文件出错：{e}')


if command ==3:
    '''
    负责按照 tap 来筛选数据的ID，目前的想法是选三个自行车出来，再选三个行人出来
    '''
    try:
        nums_I_want_to_select = 3
        path2dataset_file = "10039.csv"
        type_i_want = 'PEDESTRIAN' # type:ignore
        output_file = '10039_3_PEDESTRIAN.csv'


        
        # print(f'dataset path: {path2dataset_file}')
        raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
        colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
        raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据

        # this_type = str(type_i_want)
        found = raw_data[raw_data['type'] == type_i_want] # type: ignore
        # print(f'找到符合的一行：\n{found}\n')
        all_ids_that_we_have = found.id.unique()
        # print(f'所有id为：{all_ids_that_we_have}')

        select_modified_num_of_id = all_ids_that_we_have[0:nums_I_want_to_select]
        # print(f'我想要的前{nums_I_want_to_select}个id为：{select_modified_num_of_id}')
       

        new_data_frame = pd.DataFrame(columns=colums_we_want)

        for this_id in select_modified_num_of_id:
            tmp_df = raw_data[raw_data['id'] == this_id]
            new_data_frame = pd.concat([new_data_frame, tmp_df], ignore_index=True)

        # print(f'看看筛选出来的df：{new_data_frame}')


        try:
            new_data_frame.to_csv(output_file, index=False)
            print(f'生成文件{output_file}成功')
        
        except Exception as ex:
             print(f'保存数据出现错误：{e}')
             
            

    except Exception as e:
         print(f'筛选数据出现错误：{e}')



if command == 4:
    '''
    将几个csv文件串联起来
    '''

    # files = ['10039_3_car.csv', '10039_3_BICYCLE.csv']
    f1 = pd.read_csv('10039_3_car.csv')
    f2 = pd.read_csv('10039_3_BICYCLE.csv')
    colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
    new_data_frame = pd.DataFrame(columns=colums_we_want)

    new_data_frame = pd.concat([f1, f2], ignore_index=True)

    output_file = '10039_3_car_bike.csv'
    new_data_frame.to_csv(output_file, index=False)
    print(f'生成文件{output_file}成功')