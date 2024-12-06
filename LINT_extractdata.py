# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:02:34 2024

@author: Alex
"""
import serial, datetime
import numpy as np
import pandas as pd
import serial.tools.list_ports as stl

ports = stl.comports()
for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))
    
ser = serial.Serial()
ser.baudrate = 57600
ser.port = 'COM3'

j, N = [0, 20]

data = np.zeros((N, 2), dtype=int)

ser.open()
ser.reset_input_buffer()
initial_t = str(datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S,%f'))
while j<N:
    data[j] = np.array(ser.readline().decode("utf-8").split(';'), dtype=int)
    print(data[j], '\n')
    j+=1
ser.reset_output_buffer()
ser.close()

df = pd.DataFrame(data, columns = ['t [s]', 'Counts']).set_index(['t [s]']).to_csv(f'dataLINT_{initial_t}_N{N}.csv')
