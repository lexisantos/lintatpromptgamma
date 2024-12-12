# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:46:16 2024

@author: Alex
"""
import serial, datetime, csv, os
import numpy as np
import serial.tools.list_ports as stl
import matplotlib.pyplot as plt

def Timestamp():
    return datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S.%f')

def Show_ports():
    ports = stl.comports()
    for port, desc, hwid in sorted(ports):
        #puerto, descripcion, descripción técnica
        print("{}: {} [{}]".format(port, desc, hwid))
            
class PulseCounter:
    def __init__(self, port, baudrate = 57600):
        global ser
        ser = serial.Serial() 
        ser.baudrate = baudrate
        ser.port = port
        
    def Initialize(self):
        global ser
        try:
            if ser.isOpen():
                print(f"\n Serial in {ser.port} is already Open. ")
            else:
                print("\n Error: Serial is not Open. Opening...")
                ser.open()
                print(f"\n Serial in {ser.port} is now Open.")
        except:
            print('\n Please, verify connection.')
    
    def Reader(self, filename = 'test_csv.csv', timeout: int=20, header = ['Date', 'PC_Timestamp', 't_LINT [s]', 'Counts']):
        read_arr = np.zeros((timeout, 2), dtype=int)
        with open(filename, 'w', newline='') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(header)
            # ser.reset_input_buffer()
            read_arr[0] = np.array(ser.readline().decode("utf-8").split(';'), dtype=int) 
            t0 = read_arr[0, 0]
            read_arr[0, 0] = 0
            linecsv = [*Timestamp().split(), *read_arr[0]]
            csv_file.writerow(linecsv)
            n = 1
            while n < timeout:
                read_arr[n] = np.array(ser.readline().decode("utf-8").split(';'), dtype=int)
                read_arr[n, 0] -= t0
                linecsv = [*Timestamp().split(), *read_arr[n]]
                csv_file.writerow(linecsv)
                # print(*linecsv)
                line1.set_xdata(read_arr[:n+1, 0])
                line1.set_ydata(read_arr[:n+1, 1])
                ax.set_xlim(read_arr[:n+1, 0].min(), read_arr[:n+1, 0].max())
                ax.set_ylim(read_arr[:n+1, 1].min(), read_arr[:n+1, 1].max())
                fig.canvas.draw()
                fig.canvas.flush_events()
                n+=1
            f.close()
        ser.reset_output_buffer()
        ser.close()
        print(f'\n Data saved in {os.getcwd()}{filename}')
     
def run(title, N, puerto = 'COM3'):
    global ax, fig, line1
    global cp
    plt.ion()
    fig, ax = plt.subplots(figsize = (8,6))
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Counts')
    ax.grid(True, ls='--')
    ax.ticklabel_format(useOffset = False)
    
    line1, = ax.plot([], [], "r.-")
    
    cp = PulseCounter(puerto) 
    cp.Initialize()
    cp.Reader(filename = title, timeout = N)

Show_ports()
#El LINT es USB VID:PID=0403:6015 SER=FTXW3XZ4A

# %% 

run('test13_genfun_13kHz.csv', N = 300)
