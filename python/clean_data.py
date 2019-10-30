from utils import *
import csv

def combine_wind_tprh_nwp(wind_tprh_csv,nwp_csv, wind_tprh_csv_out):
    wind_tprh_f = open(wind_tprh_csv, 'r')
    nwp_csv_f = open(nwp_csv, 'r')
    wind_tprh_csv_f = open(wind_tprh_csv_out, 'w')
    
    wind_tprh_c = csv.reader(wind_tprh_f)
    nwp_csv_c = csv.reader(nwp_csv_f)
    wind_tprh_csv_c = csv.writer(wind_tprh_csv_f)

    wind_tprh_list = list(wind_tprh_c)
    nwp_list = list(nwp_csv_c)
    

    wind_tprh_data = [['data-time','wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp']]
    i = 1
    while i < len(wind_tprh_list):
        wind_tprh_info = wind_tprh_list[i]
        nwp_info = nwp_list[i][1:]
        print('dealing',wind_tprh_info,nwp_info)
        
        tmp = []
        tmp.extend(wind_tprh_info)
        tmp.extend(nwp_info)

        wind_tprh_data.append(tmp)
        i = i + 1
    
    wind_tprh_csv_c.writerows(wind_tprh_data)

    wind_tprh_f.close()
    nwp_csv_f.close()
    wind_tprh_csv_f.close()

    pass

def combine_wind_tprh(TPRH_CSV,WIND_CSV, wind_tprh_out):
    wind_f = open(WIND_CSV, 'r')
    tprh_f = open(TPRH_CSV, 'r')
    wind_tprh_f = open(wind_tprh_out, 'w')
    
    wind_c = csv.reader(wind_f)
    tprh_c = csv.reader(tprh_f)
    wind_tprh_c = csv.writer(wind_tprh_f)

    wind_list = list(wind_c)
    tprh_list = list(tprh_c)
    

    wind_tprh_data = [['data-time','wind','ap','tmp','humi']]

    i = 1
    while i < len(wind_list):
        wind_info = wind_list[i]
        tprh_info = tprh_list[i]
        print('dealing',wind_info,tprh_info)
        
        dataTime = wind_info[0]
        wind = wind_info[1]
        ap = tprh_info[1]
        tmp = tprh_info[2]
        humi = tprh_info[3]

        wind_tprh_data.append([dataTime,wind,ap,tmp,humi])
        i = i + 1
    
    wind_tprh_c.writerows(wind_tprh_data)

    wind_f.close()
    tprh_f.close()
    wind_tprh_f.close()


def clean_tprh_data(tprh_raw,tprh_out):
    tprh_f = open(tprh_raw, 'r')
    date_data = []
    time_data = []
    tmp_data = []
    ap_data = [] #air_pressure
    humi_data = []
    
    for info in tprh_f:
        print('dealing', info)
        info = info.split()
        
        date_data.append(info[0])
        time_data.append(info[1])
        ap = float(info[2])
        tmp = float(info[3])
        if len(info) > 4:
            humi = float(info[4])
        else:
            humi = 0
        ap_data.append(ap)        
        tmp_data.append(tmp)
        humi_data.append(humi)


    #fill wind abnormal with difference

    TPRH_CSV_data = [['date-time','ap','tmp','humi']]    
    for i in range(len(date_data)): 
        
        date_time  = date_data[i]+'-'+time_data[i]
        tmp = tmp_data[i]
        ap = ap_data[i]
        humi = humi_data[i]
        
        TPRH_CSV_data.append([date_time,ap,tmp,humi])
    
    with open(tprh_out, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(TPRH_CSV_data)

def clean_wind_data(wind_raw,wind_out):
    wind_f = open(wind_raw, 'r')
    date_data = []
    time_data = []
    wind_data = []


    for wind in wind_f:
        wind = wind.split()
        print(wind)
        date_data.append(wind[0])
        time_data.append(wind[1])
        if len(wind) > 2:
            wind = float(wind[2])
        else:
            wind = -1
        wind_data.append(wind)

    #fill wind abnormal with difference

    WIND_CSV_data = [['date-time','wind']]    
    for i in range(len(wind_data)):  
        date_time  = date_data[i]+'-'+time_data[i]
        print('processing',date_time)
        wind = wind_data[i]
        if wind < WIND_MIN or wind > WIND_MAX:
            prev = i - 1
            after = i + 1
            wind_prev = 0
            wind_after = 0
            while prev >= 0:
                if wind_data[prev] >= WIND_MIN and wind_data[prev] <= WIND_MAX:
                    wind_prev = wind_data[prev]
                    break
            while after < len(wind_data):
                if wind_data[after] >= WIND_MIN and wind_data[after] <= WIND_MAX:
                    wind_after = wind_data[after]
                    break
            
            wind_data[i] = (wind_after + wind_prev)/2
        
        WIND_CSV_data.append([date_time,wind])

    with open(wind_out, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(WIND_CSV_data)

def getNwpdate(date_str):
    date = int(date_str)
    date = date / 100
    return int(date)


def clean_nwp_data(nwp_raw,nwp_out):
    nwp_data = []
    with open(nwp_raw, 'r') as nwp_f:
        for i in nwp_f:
            nwp_data.append(i)
    
    row = 0     
    NWP_CSV_data = [['data-time','wind','dir','u','v', 't', 'rh', 'psfc','slp']]
    while row < len(nwp_data):         
        
        for hour in range(24):

            date = ''
            if row < len(nwp_data) - 8:
                date = getNwpdate(nwp_data[row + 8].split()[0])
            else:
                date = getNwpdate(nwp_data[row].split()[0]) + 1

            nwp_row_data = [str(date)+'-'+str(hour)]                    
            
            offset = 0
            while offset < 8:
                
                current_row = row + offset
                if row < len(nwp_data) - 8:
                    speed_info0 = nwp_data[current_row].split()
                    speed_info1 = nwp_data[current_row + 8].split() 
                    nwp_yesterday36_hours = speed_info0[2:]
                    nwp_today36_hours = speed_info1[2:]
                else:
                    speed_info0 = nwp_data[current_row].split()
                    nwp_yesterday36_hours = speed_info0[2:]
                    
              
                value = 0
              

                if row < len(nwp_data) - 8:
                    if hour <= 11:
                        print("processing", date,hour  )
                        value = float(nwp_yesterday36_hours[12 + hour])
                    else:                       
                        value = float(nwp_today36_hours[hour - 12])
                        
                else: #the last day                    
                    value = float(nwp_yesterday36_hours[12 + hour])
                
                nwp_row_data.append(value)


                offset += 1
            
            NWP_CSV_data.append(nwp_row_data)        
        row += 8        

    with open(nwp_out, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(NWP_CSV_data)
    pass


if __name__ == '__main__':
    #clean_wind_data(DATAPATH+WIND_TXT,  DATAPATH+WIND_CSV)
    #clean_nwp_data(DATAPATH+NWP_TXT, DATAPATH+NWP_CSV)
    #clean_tprh_data(DATAPATH + TPRH_TXT, DATAPATH + TPRH_CSV)
    #combine_wind_tprh(DATAPATH +TPRH_CSV,DATAPATH + WIND_CSV, DATAPATH + WIND_TPRH_CSV)
    combine_wind_tprh_nwp(DATAPATH +WIND_TPRH_CSV,DATAPATH + NWP_CSV, DATAPATH + WIND_TPRH_NWP_CSV)


